from __future__ import absolute_import
import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 20
ignore_label = 255
root = '/root/Share/LIP'

'''
color map
0=background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 5=Dress, 6=Coat, 7=Socks, 8=Pants, 9=Glove, 10=Scarf, 11=Skirt,
12=Jumpsuits, 13=Face, 14=Right-arm, 15=Left-arm # 16=Right-leg, 17=Left-leg, 18=Right-shoe, 19=Left-shoe
'''

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode != 'test':
        img_path = os.path.join(root, 'TrainVal_images/TrainVal_images', '{}_images'.format(mode))
        mask_path = os.path.join(root, 'TrainVal_parsing_annotations','{}_segmentations'.format(mode))
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'TrainVal_images', '{}_id.txt'.format(mode))).readlines()]
        for it in data_list[:40]:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


class LIP(data.Dataset):
    def __init__(self, mode, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        # self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
