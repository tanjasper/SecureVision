import torchvision.models as models
import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from fns_top5acc import train, validate, AverageMeter, accuracy
import os

# some parameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', default=4, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--checkpoint_path', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('--proj_type', default=0, type=int)
# Code:
#   0 -- no projection
#   1 -- projection by fixed random matrix then transpose

def main():

    # call up parameters
    opt = parser.parse_args()
    start_epoch = 0

    # create save directory if it doesn't exist
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    # first I need my model (and placed in GPU)
    model = models.alexnet()
    torch.cuda.set_device(opt.gpu_idx)
    model = model.cuda(opt.gpu_idx)

    # next, we build our random projection matrix (as pytorch tensor)
    if opt.proj_type == 0:
        proj_step = None
    elif opt.proj_type == 1:
        norm_mean = 0
        norm_std = 1
        imdim = 224*224
        projdim = round(0.1 * imdim)
        proj_step = proj_projT(imdim, projdim, norm_mean=norm_mean, norm_std=norm_std)
        proj_step.cuda(opt.gpu_idx)

    # data shit 1: datasets
    traindir = '/media/hdd2/Datasets/ILSVRC2012/images/original/train'
    valdir = '/media/hdd2/Datasets/ILSVRC2012/images/original/val_with_subdir'
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    # data shit 2: dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda(opt.gpu_idx)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # define loss-tracker
    accs = {'tr1': [], 'tr5': [], 'val1': [], 'val5': [], 'best_val_acc1': 0}

    # load checkpoint if ever
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if proj_step is not None:
            proj_step.load_state_dict(checkpoint['proj_step_state_dict'])
        accs = checkpoint['accs']
        start_epoch = checkpoint['last_finished_epoch'] + 1

    # train!
    for epoch in range(start_epoch, opt.num_epochs):

        # adjust learning rate
        curr_lr = opt.lr * 0.1**(epoch // 10)  # decrease learning rate by 0.1 every 10 epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        # train for one epoch
        train(train_loader, model, proj_step, criterion, optimizer, epoch, accs, opt)

        # evaluate on validation set
        val_acc1 = validate(val_loader, model, proj_step, criterion, accs, opt)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > accs['best_val_acc1']
        accs['best_val_acc1'] = max(val_acc1, accs['best_val_acc1'])
        if proj_step is not None:
            proj_step_state_dict = proj_step.state_dict()
        else:
            proj_step_state_dict = None
        save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'proj_step_state_dict': proj_step_state_dict,
                'accs': accs,
                'last_finished_epoch': epoch
            }, is_best, opt.save_dir)


#  generate a random Gaussian matrix and its transpose
class proj_projT(nn.Module):
    def __init__(self, imdim, projdim, norm_mean=0, norm_std=1):
        super(proj_projT, self).__init__()

        # generate matrix and its transpose
        proj_mat = torch.zeros([projdim, imdim])
        proj_mat.normal_(norm_mean, norm_std)
        proj_matT = proj_mat.transpose(1, 0)

        # convert projection matrix into CUDA pytorch modules
        self.proj = nn.Linear(imdim, projdim, bias=False)
        self.projT = nn.Linear(projdim, imdim, bias=False)
        self.proj.weight = nn.Parameter(proj_mat)
        self.projT.weight = nn.Parameter(proj_matT)
        self.proj.weight.requires_grad = False  # fix these matrices
        self.projT.weight.requires_grad = False

    def forward(self, x):
        for i in range(x.size(1)):  # project each channel
            tmp = x[:, i, :, :].view(x.size(0), -1)
            tmp = self.proj(tmp)
            tmp = self.projT(tmp)
            tmp = tmp.view(x.size(0), x.size(2), x.size(3))
            x[:, i, :, :] = tmp
        return x

def save_checkpoint(save_dict, is_best, save_dir):
    # first, save the network as the latest network
    torch.save(save_dict, os.path.join(save_dir, 'latest_network.tar'))
    # if it is the best network, replace old best network
    if is_best:
        torch.save(save_dict, os.path.join(save_dir, 'best_network.tar'))


if __name__ == '__main__':
    main()
