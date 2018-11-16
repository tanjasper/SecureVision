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

    # Options
    global opt
    opt = argparse.ArgumentParser(description='FlatCam Face Verification')
    opt.add_argument('--data_dir', default='/media/hdd1/Datasets/casia/two_subjects_images')
    opt.add_argument('--filenames', default='/media/hdd1/Datasets/casia/filenames_two_subjects.txt')
    opt.add_argument('-j', '--num_threads', default=4, type=int, help='number of data loading workers (default: 4)')
    opt.add_argument('--seed', default=1313, type=int, help='seed for initializing training (default: 1313)')
    opt.add_argument('--gpu', '--gpu_ids', default=0, type=int, help='GPU id to use (default=0)')
    opt.add_argument('--model_path', default=None, required=True)
    opt.add_argument('--save_path', default=None, required=True)
    opt = opt.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        # cudnn.deterministic = True  # might slow things down, but eliminates randomness for reproducibility
        cudnn.deterministic = True

    # Load model
    model = networks.MaskAlexNet()
    model = model.cuda(opt.gpu)
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage.cuda(0))
    print('Done with torch.load')
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    # Prepare data
    # datasetA = datasets.DatasetFromFilenames(opt, opt.dataA_dir, opt.filenamesA)
    # datasetB = datasets.DatasetFromFilenames(opt, opt.dataB_dir, opt.filenamesB)
    dataset = my_datasets.ImageFolder(
        opt.data_dir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize([255, 255], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    # create a dataloader where batch_size is entire dataset (so one sample gives all data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                          num_workers=opt.num_threads, pin_memory=True)

    # obtain the images
    input = next(iter(loader))
    labels = input[1]
    input = input[0]
    if opt.gpu is not None:
        input = input.cuda(opt.gpu, non_blocking=True)
    input = input[:, 0, :, :].unsqueeze(1)  # send in red channel only

    # send through network
    feats = model.optics(input)  # obtain sensor measurements
    feats = feats.view(feats.shape[0], -1)  # convert feats to be [batch_size, feat_size]
    feats = feats.cpu()  # convert to cpu?

    # normalize to be from 0 to 1
    # feats = (feats + feats.min(dim=0)[0]) / (feats.max(dim=0)[0] - feats.min(dim=0)[0])
    feats_temp = feats.unsqueeze(1)

    # Maybe we should do everything in numpy
    feats = feats.detach().numpy()
    feats_temp = feats_temp.detach().numpy()
    distance_matrix = np.zeros((feats.shape[0], feats.shape[0]))
    for i in range(feats.shape[0]):
        if i % 50 == 0:
            print('Done with subject %d out of %d' % (i, feats.shape[0] - 1))
        tmp = abs((feats_temp - feats[i]).squeeze()).max(1)  # l_inf diff of feat i with every feat
        distance_matrix[:, i] = tmp  # inf norm of each difference


    # # calculate distances (l_inf norm)
    # distance_matrix = torch.zeros(feats.shape[0], feats.shape[0])
    # for i in range(feats.shape[0]):
    #     if i % 50 == 0:
    #         print('Done with subject %d out of %d' % (i, feats.shape[0]-1))
    #     tmp = torch.max(torch.abs((feats_temp - feats[i]).squeeze()), 1)[0]  # l_inf diff of feat i with every feat
    #     distance_matrix[:, i] = tmp  # inf norm of each difference
    # distance_matrix = distance_matrix.numpy()
    labels = labels.numpy()

    np.savez(opt.save_path, distance_matrix=distance_matrix, labels=labels)


if __name__ == '__main__':
    main()