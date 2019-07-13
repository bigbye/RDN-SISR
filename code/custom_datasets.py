import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image as Image
import random
import torch


def random_crop(lr, hr, size, scale):
    lr_left = random.randint(0, lr.shape[1] - size)
    lr_right = lr_left + size
    lr_top = random.randint(0, lr.shape[0] - size)
    lr_bottom = lr_top + size
    hr_left = lr_left * scale
    hr_right = lr_right * scale
    hr_top = lr_top * scale
    hr_bottom = lr_bottom * scale
    lr = lr[lr_top:lr_bottom, lr_left:lr_right]
    hr = hr[hr_top:hr_bottom, hr_left:hr_right]
    return lr, hr


def augment(lr, hr):
    # random_vertical_flip
    if random.random() < 0.5:
        lr = lr[::-1, :, :].copy()
        hr = hr[::-1, :, :].copy()

    # random_rotate_90
    if random.random() < 0.5:
        lr = np.rot90(lr, axes=(1, 0)).copy()
        hr = np.rot90(hr, axes=(1, 0)).copy()

    # random_horizontal_flip
    if random.random() < 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()
    return lr, hr


def to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
    img = torch.div(img, 255.0)
    return img


def get_train_list(lr_path, hr_path):
    lr_list = sorted(os.listdir(lr_path))
    hr_list = sorted(os.listdir(hr_path))

    lr_path_list = [os.path.join(lr_path, str(i)) for i in lr_list]
    hr_path_list = [os.path.join(hr_path, str(i)) for i in hr_list]

    return lr_path_list, hr_path_list


def get_test_list(lr_path):
    lr_list = sorted(os.listdir(lr_path))
    lr_path_list = [os.path.join(lr_path, str(i)) for i in lr_list]
    return lr_path_list


class TrainDataset(Dataset):
    def __init__(self, lr_train_path, hr_train_path, patch_size, scale, is_train=True):
        super(TrainDataset, self).__init__()

        self.lr_train_path = lr_train_path
        self.hr_train_path = hr_train_path
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train
        self.lr_path_list, self.hr_path_list = get_train_list(lr_train_path, hr_train_path)

    def __getitem__(self, idx):
        lr = np.array(Image.open(self.lr_path_list[idx]).convert('RGB'))
        hr = np.array(Image.open(self.hr_path_list[idx]).convert('RGB'))  # shape hwc
        if self.is_train:
            lr, hr = random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = augment(lr, hr)
        return to_tensor(lr), to_tensor(hr)

    def __len__(self):
        return len(self.lr_path_list)

    def __str__(self):
        return 'custom_train_dataset'


class TestDataset(Dataset):
    def __init__(self, lr_test_path):
        super(TestDataset, self).__init__()
        self.lr_path_list = get_test_list(lr_test_path)

    def __getitem__(self, idx):
        lr_path = self.lr_path_list[idx]
        lr = np.array(Image.open(lr_path).convert('RGB'))
        return idx, to_tensor(lr), lr_path

    def __len__(self):
        return len(self.lr_path_list)

    def __str__(self):
        return 'custom_test_dataset'
