import argparse
import os
import random
import shutil
import time
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import options_direct_inference
import losses
import networks
import data
import my_datasets

best_acc1 = 0


def main():
    global lossvals
    opt = options_direct_inference.generate_parser()

    start_epoch = 0
    im_dim = opt.mask_size + 224 - 1

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic = True

    model = networks.MaskVGG16(num_classes=2, batch_norm=True, mask_size=opt.mask_size, mask_perc=0.5)
    model = model.cuda(opt.gpu)

    criterion = nn.CrossEntropyLoss().cuda(opt.gpu)

    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    lossvals = dict()
    lossvals['train'] = opt.epochs*[0]
    lossvals['train_class0'] = opt.epochs * [0]
    lossvals['train_class1'] = opt.epochs * [0]
    lossvals['val'] = opt.epochs * [0]
    lossvals['val_class0'] = opt.epochs * [0]
    lossvals['val_class1'] = opt.epochs * [0]

    # optionally resume from a checkpoint
    if opt.resume_epoch is not None:
        resume_filename = os.path.join(opt.checkpoints_dir, 'checkpoint_epoch%d.pth.tar' % opt.resume_epoch)
        if os.path.isfile(resume_filename):
            print("=> loading checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename, map_location=lambda storage, loc: storage.cuda(0))
            print('Done with torch.load')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            curr_lr = optimizer.param_groups[0]['lr']
            #lossvals = np.load(os.path.join(opt.checkpoints_dir, 'lossvals.npy'))
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_filename))

    # freeze mask weights if needed
    if not opt.train_mask:
        for param in model.optics.parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    cudnn.benchmark = True

    # Load training data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(im_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
    ])
    train_dataset = data.DatasetFromMultipleFilenames(opt,
                                                      [opt.dataA_dir, opt.dataB_dir],
                                                      [opt.filenamesA, opt.filenamesB],
                                                      train_transform)

    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize(im_dim+32),
        transforms.CenterCrop(im_dim),
        transforms.ToTensor(),
        #normalize,
    ])
    val_dataset = data.DatasetFromMultipleFilenames(opt,
                                                    [opt.dataA_dir, opt.dataB_dir],
                                                    [opt.filenamesA_val, opt.filenamesB_val],
                                                    val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)

    if opt.evaluate:
        validate(val_loader, model, criterion, 0, opt)
        return

    for epoch in range(start_epoch, opt.epochs):

        adjust_learning_rate(optimizer, epoch, opt)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, opt)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        np.save(os.path.join(opt.checkpoints_dir, 'lossvals.npy'), lossvals)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accmeter = AverageMeter()
    accmeter_class0 = AverageMeter()
    accmeter_class1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        output = model(input, opt)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc, acc_class0, acc_class1 = binary_accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        accmeter.update(acc[0], input.size(0))
        accmeter_class0.update(acc_class0[0], (target == 0).sum().cpu().item())
        accmeter_class1.update(acc_class1[0], (target == 1).sum().cpu().item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc (class 0) {class0.val:.3f} ({class0.avg:.3f})\t'
                  'Acc (class 1) {class1.val:.3f} ({class1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=losses, top1=accmeter, class0=accmeter_class0, class1=accmeter_class1))

    lossvals['train'][epoch] = accmeter.avg
    lossvals['train_class0'][epoch] = accmeter_class0.avg
    lossvals['train_class1'][epoch] = accmeter_class1.avg


def validate(val_loader, model, criterion, epoch, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accmeter = AverageMeter()
    accmeter_class0 = AverageMeter()
    accmeter_class1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            output = model(input, opt)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, acc_class0, acc_class1 = binary_accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            accmeter.update(acc[0], input.size(0))
            accmeter_class0.update(acc_class0[0], (target == 0).sum().cpu().item())
            accmeter_class1.update(acc_class1[0], (target == 1).sum().cpu().item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc (class 0) {class0.val:.3f} ({class0.avg:.3f})\t'
                      'Acc (class 1) {class1.val:.3f} ({class1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=accmeter, class0=accmeter_class0, class1=accmeter_class1))

        print(' * Acc {top1.avg:.3f} Acc (class 0) {class0.avg:.3f} Acc (class 1) {class1.avg:.3f}'
              .format(top1=accmeter, class0=accmeter_class0, class1=accmeter_class1))

    lossvals['val'][epoch] = accmeter.avg
    lossvals['val_class0'][epoch] = accmeter_class0.avg
    lossvals['val_class1'][epoch] = accmeter_class1.avg

    return accmeter.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def binary_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.max(output.data, 1)
        correct = pred.eq(target)
        res = []
        res.append(correct.float().sum(0, keepdim=True).mul_(100.0 / batch_size))
        # find out per class
        idx_class0 = (target == 0)
        idx_class1 = (target == 1)
        res.append(correct[idx_class0].float().sum(0, keepdim=True).mul_(100.0 / idx_class0.float().sum()))
        res.append(correct[idx_class1].float().sum(0, keepdim=True).mul_(100.0 / idx_class1.float().sum()))

        return res


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()