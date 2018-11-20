import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image

import options
import losses
import networks
import data
import my_datasets

def main():
    global opt, closest_pairs, epoch, lossvals
    start_epoch = 0
    opt = options.generate_parser()

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        # cudnn.deterministic = True  # might slow things down, but eliminates randomness for reproducibility
        cudnn.deterministic = True

    # Load or create model
    model = networks.MaskAlexNet()
    model = model.cuda(opt.gpu)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)

    # define loss function
    loss_fn = losses.HardNegativeLoss()

    # create checkpoints folder
    if not os.path.exists(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)

    if opt.resume_epoch is not None:
        resume_filename = os.path.join(opt.checkpoints_dir, 'checkpoint_epoch%d.pth.tar' % opt.resume_epoch)
        if os.path.isfile(resume_filename):
            print("=> loading checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename, map_location=lambda storage, loc: storage.cuda(0))
            print('Done with torch.load')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_filename))

    cudnn.benchmark = True

    # Prepare data
    # datasetA = datasets.DatasetFromFilenames(opt, opt.dataA_dir, opt.filenamesA)
    # datasetB = datasets.DatasetFromFilenames(opt, opt.dataB_dir, opt.filenamesB)

    # difference between this and regular ImageFolder is that this also returns index of image
    datasetA = my_datasets.ImageFolder(
        opt.dataA_dir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize([351, 351], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    # datasetBoth = data.MultipleImageFolder(
    #     [opt.dataA_dir, opt.dataB_dir],
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(224),
    #         transforms.Resize([255, 255], interpolation=Image.BICUBIC),
    #         transforms.ToTensor(),
    #         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]))

    loaderA = torch.utils.data.DataLoader(
        datasetA, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    # loaderB = torch.utils.data.DataLoader(
    #    datasetB, batch_size=opt.batch_size, num_workers=opt.num_threads, pin_memory=True)

    # loaderBoth = torch.utils.data.DataLoader(
    #     datasetBoth, batch_size=opt.batch_sizeB, num_workers=opt.num_threads, pin_memory=True)

    if os.path.isfile(os.path.join(opt.checkpoints_dir, 'closest_pairs.npy')):
        closest_pairs = np.load(os.path.join(opt.checkpoints_dir, 'closest_pairs.npy'))
    else:
        closest_pairs = np.zeros((len(datasetA), opt.epochs)).astype(int)

    # keep track of loss values
    lossvals = []
    curr_lr = opt.lr
    iter_this_lr = 0  # iterations at this learning rate

    for epoch in range(start_epoch, opt.epochs):

        # decrease learning rate after 50 epochs
        #lr = opt.lr * (0.1 ** (epoch // 50))
        #lr = opt.lr
        # decrease learning rate if training error is not decreasing
        if epoch > 1 and lossvals[epoch-1] > lossvals[epoch-2] and iter_this_lr > 5:
            curr_lr = curr_lr * 0.1
            iter_this_lr = 0
        print('Learning rate is %0.6f' % curr_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        iter_this_lr = iter_this_lr + 1

        # train for one epoch
        train(loaderA, model, loss_fn, optimizer, epoch)

        # save checkpoint
        if epoch % opt.save_freq == 0:
            save_checkpoint(opt, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch)
            np.save(os.path.join(opt.checkpoints_dir, 'closest_pairs.npy'), closest_pairs)
            mask = model.optics.weight.cpu().detach().numpy().squeeze()
            np.save(os.path.join(opt.checkpoints_dir, 'mask_epoch%d.npy' % epoch), mask)

    print('Minimum loss is %f after epoch %d' % (np.min(lossvals), np.argmin(lossvals)-1))

def train(train_loader, model, loss_fn, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    lossmeter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    temp_lossvals = []

    for i, (input, target, idx) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # clamp mask weights to be non-negative (maybe?)
        if opt.nn_mask:
            model.optics.weight.data.clamp_(min=0, max=1)

        # compute output
        input = input[:, 0, :, :].unsqueeze(1)  # send in red channel only
        output = model.optics(input)  # obtain sensor measurements
        loss_outputs = loss_fn(output, target, idx)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        closest_pairs[idx, epoch] = loss_outputs[1]  # supposedly, loss_outputs[1] is the closest_pairs
        temp_lossvals.append(loss.item())
        lossmeter.update(loss.item(), input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=lossmeter))

    lossvals.append(np.mean(temp_lossvals))


def save_checkpoint(opt, state, epoch):
    temp_filename = 'checkpoint_epoch%d.pth.tar' % epoch
    filename = os.path.join(opt.checkpoints_dir, temp_filename)
    torch.save(state, filename)


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


if __name__ == '__main__':
    main()
