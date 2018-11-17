import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import torch.utils.data as data
from torchvision.datasets.folder import make_dataset
import math
import re
import sys
import torch


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

# DatasetFolder that takes in multiple roots. Each root pertains to one class
# Modified from torchvision.datasets.DatasetFolder
class MultipleDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, roots, loader, extensions, transform=None, target_transform=None):

        samples = []
        root_lengths = []

        for i in range(len(roots)):
            root = roots[i]
            classes, class_to_idx = self._find_classes(root)
            temp_samples = make_dataset(root, class_to_idx, extensions)  # should be the path names of the images
            samples = samples + temp_samples
            root_lengths.append(len(samples))
            if len(samples) == 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                   "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        #self.targets = [s[1] for s in samples]
        self.targets = [i for (i, s) in enumerate(root_lengths) for a in range(s)]
        # Above line makes a list of idxs for each image where the idx is the root idx
        # s is the root_length, i is the root_idx. For each s, repeat idx i s times.

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = self.targets[index]  # override DatasetFolder's target
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MultipleImageFolder(MultipleDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader):
        super(MultipleImageFolder, self).__init__(roots, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


# copy-pasted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder or dataset_type is MultipleImageFolder:  # i changed this
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples