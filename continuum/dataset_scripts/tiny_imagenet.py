import pickle
import numpy as np
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
# from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
# from .concat import ConcatDataset


TEST_SPLIT = 1 / 6


class Tiny_ImageNet(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'mini_imagenet'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
            self.base_class = params.base_class
        super(Tiny_ImageNet, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)


    def download_load(self):

        def TinyImageNetLoader(batch_size, num_workers=4, path='./datasets/tiny-imagenet-200/', aug=None, shuffle=False,
                               class_list=range(150), subfolder='train'):
            dataset = TinyImageNet200(aug=aug, subfolder=subfolder, class_list=class_list, path=path)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                    pin_memory=True)
            return dataloader

        train_in = open("datasets/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open("datasets/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open("datasets/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            rdm_x, rdm_y = shuffle_data(cur_x, cur_y)
            x_test = rdm_x[: int(600 * TEST_SPLIT)]
            y_test = rdm_y[: int(600 * TEST_SPLIT)]
            x_train = rdm_x[int(600 * TEST_SPLIT):]
            y_train = rdm_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        self.train_data = np.concatenate(train_data)
        self.train_label = np.concatenate(train_label)
        self.test_data = np.concatenate(test_data)
        self.test_label = np.concatenate(test_label)

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        else:
            raise Exception('unrecognized scenario')
        return x_train, y_train, labels

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 84,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)

        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=100, base_class=self.base_class, num_tasks=self.task_nums,
                                                       fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                self.params.ns_factor)




def find_classes_from_folder(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def find_classes_from_file(file_path):
    with open(file_path) as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)

    return samples


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):

        if len(samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders \n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)


def TinyImageNet200(aug=None, subfolder='train', class_list=range(150), path='./datasets/tiny-imagenet-200/'):
    # img_split = 'images/'+subfolder
    img_split = subfolder
    classes_200, class_to_idx_200 = find_classes_from_file(os.path.join(path, 'tinyimagenet_200.txt'))

    classes_sel = [classes_200[i] for i in class_list]

    samples = make_dataset(path + img_split, classes_sel, class_to_idx_200)

    if aug == None:
        transform = transforms.Compose([
            transforms.Resize(64),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = ImageFolder(transform=transform, samples=samples)
    return dataset


def TinyImageNetLoader(batch_size, num_workers=4, path='./datasets/tiny-imagenet-200/', aug=None, shuffle=False,
                       class_list=range(150), subfolder='train'):
    dataset = TinyImageNet200(aug=aug, subfolder=subfolder, class_list=class_list, path=path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

