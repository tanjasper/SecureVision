import torchvision.models as models
import argparse
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fns import train, validate, AverageMeter, accuracy
import os
import data_fns

# some parameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', default=4, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--gpu_idx', default=0, type=int)
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=90, type=int)
parser.add_argument('--checkpoint_path', default='')
parser.add_argument('--proj_mat_path', default='')
parser.add_argument('--save_dir', default='')
parser.add_argument('--noise_var', default=0, type=float)
parser.add_argument('--proj_type', default=0, type=int)
parser.add_argument('--measurement_rate', default=0.25, type=float)
parser.add_argument('--decrease_lr_freq', default=30, type=int)
# data parameters
parser.add_argument('--face_dir', default='/media/jasper/Samsung_T5/vggface2')
parser.add_argument('--tr_filenames', default='/media/hdd2/Datasets/vggface2/filenames/most_popular_10_training_filenames.txt')
parser.add_argument('--val_filenames', default='/media/hdd2/Datasets/vggface2/filenames/most_popular_10_val_filenames.txt')
# Code:
#   0 -- no projection
#   1 -- projection by fixed random matrix then transpose

def main():

    # call up parameters
    opt = parser.parse_args()
    start_epoch = 0

    save_dir = os.path.join('/media/hdd2/research/privacy_pytorch/direct_inference_lowrank/fixed_matrix_transpose_face_identification', opt.save_dir)

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # first I need my model (and placed in GPU)
    model = models.alexnet(num_classes=10)
    torch.cuda.set_device(opt.gpu_idx)
    model = model.cuda(opt.gpu_idx)

    # next, we build our random projection matrix (as pytorch tensor)
    if opt.proj_type == 0:
        proj_step = None
    elif opt.proj_type == 1:
        norm_mean = 0
        norm_std = 1
        imdim = 224*224
        projdim = round(opt.measurement_rate * imdim)
        proj_mat = np.random.normal(norm_mean, norm_std, (projdim, imdim))
        proj_step = proj_projT(imdim, projdim, proj_mat=proj_mat)
        proj_step.cuda(opt.gpu_idx)

    # data shit 1: datasets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_dataset = data_fns.DatasetFromFilenames(
        opt, opt.face_dir, opt.tr_filenames, train_transforms)
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_dataset = data_fns.DatasetFromFilenames(
       opt, opt.face_dir, opt.val_filenames, val_transforms)

    # data shit 2: dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads, pin_memory=True)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda(opt.gpu_idx)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # define loss-tracker
    accs = {'tr': [], 'val': [], 'best_val_acc': 0}

    # load checkpoint if ever
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # load the projection matrix used
        if proj_step is not None:
            if opt.proj_mat_path is None:  # projection matrix not provided, it is in save directory
                proj_mat_path = os.path.join(save_dir, 'proj_mat.npy')
            proj_mat = np.load(proj_mat_path)
            proj_step = proj_projT(imdim, projdim, proj_mat=proj_mat)
            proj_step.cuda(opt.gpu_idx)
        accs = checkpoint['accs']
        start_epoch = checkpoint['last_finished_epoch'] + 1
    else:  # else, save the new projection matrix
        print('Saving projection matrix')
        np.save(os.path.join(save_dir, 'proj_mat.npy'), proj_mat)
        print('Done saving projection matrix')

    # train!
    for epoch in range(start_epoch, opt.num_epochs):

        # adjust learning rate
        curr_lr = opt.lr * 0.1**(epoch // opt.decrease_lr_freq)  # decrease learning rate by 0.1 every 10 epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        # train for one epoch
        train(train_loader, model, proj_step, criterion, optimizer, epoch, accs, opt)

        # evaluate on validation set
        val_acc1 = validate(val_loader, model, proj_step, criterion, accs, opt)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > accs['best_val_acc']
        accs['best_val_acc'] = max(val_acc1, accs['best_val_acc'])
        #if proj_step is not None:
        #    proj_step_state_dict = proj_step.state_dict()
        #else:
        #    proj_step_state_dict = None
        save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'proj_step_state_dict': proj_step_state_dict,
                'accs': accs,
                'last_finished_epoch': epoch
            }, is_best, save_dir)

        # plot and print
        f = plt.figure()
        xp = range(len(accs['tr']))
        plt.plot(xp, [z / 100 for z in accs['tr']], xp, [z / 100 for z in accs['val']])
        plt.legend(('tr', 'val'))
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        f.savefig(os.path.join(save_dir, 'acc_plots.pdf'), bbox_inches='tight')


#  generate a random Gaussian matrix and its transpose
# proj_mat is a Numpy tensor of the linear projection matrix
class proj_projT(nn.Module):
    def __init__(self, imdim, projdim, proj_mat=None, norm_mean=0, norm_std=1):
        super(proj_projT, self).__init__()

        # generate matrix and its transpose
        if proj_mat is None:
            proj_mat = torch.zeros([projdim, imdim])
            proj_mat.normal_(norm_mean, norm_std)
        else:
            proj_mat = torch.Tensor(proj_mat)
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
