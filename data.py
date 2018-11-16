import torchvision.transforms as transforms
from PIL import Image
import os
import math
import re


# Class for loading data from a single dataset with a single filenames text
class DatasetFromFilenames:

    def __init__(self, opt, data_dir, filenames_loc):
        self.opt = opt
        self.data_dir = data_dir
        self.filenames = filenames_loc
        self.paths = get_paths(self.filenames, self.data_dir)
        self.num_im = len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        print('Index is %d' % index)
        # obtain the image paths
        im_path = self.paths[index % self.num_im]
        # load images (grayscale for direct inference)
        im = Image.open(im_path).convert('RGB')
        # perform transformations
        im = transforms.Resize([255, 255], interpolation=Image.BICUBIC)(im)
        im = transforms.ToTensor()(im)
        im = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im)  # imagenet normalization
        # im = transforms.Normalize((0.588, 0.4489, 0.3844), (0.2789, 0.2426, 0.3089))(im)
        return im


# Class for loading data from two datasets with two filenames texts
class DatasetFromTwoFilenames:

    def __init__(self, opt, data_dir, filenames_loc):
        self.opt = opt
        self.data_dir = data_dir
        self.filenames = filenames_loc
        self.paths = get_paths(self.filenames, self.data_dir)
        self.num_im = len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # obtain the image paths
        im_path = self.paths[index % self.num_im]
        # load images (grayscale for direct inference)
        im = Image.open(im_path).convert('RGB')
        # perform transformations
        im = transforms.Resize([255, 255], interpolation=Image.BICUBIC)(im)
        im = transforms.ToTensor()(im)
        im = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(im)  # imagenet normalization
        # im = transforms.Normalize((0.588, 0.4489, 0.3844), (0.2789, 0.2426, 0.3089))(im)
        return im


# function for getting image paths out from filenames text file
def get_paths(fname, dir=''):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            temp = str(line).strip()
            paths.append(os.path.join(dir, temp))
    return paths
