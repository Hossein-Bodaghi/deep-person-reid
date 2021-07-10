#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:34:00 2021

@author: hossein
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from delivery import data_delivery 
from loaders import MarketLoader
#%%
main_path = '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/market1501_label/final_attr.npy'
path_start = '/home/hossein/market1501_label/final_stop.npy'
# loading attributes

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     path_start=path_start,
                     need_attr=True)


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


transform1 = transforms.Compose([transforms.RandomRotation(degrees=15),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomPerspective(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# split data to test and train
train_idx, val_idx = train_test_split(list(range(len(attr['img_names']))),
                                      test_size=0.05)

train_data = MarketLoader(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=transform1,
                          indexes=train_idx)
 
test_data = MarketLoader(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=transform2,
                          indexes=val_idx) 

train_loader = DataLoader(train_data,batch_size=5,shuffle=True)
test_loader = DataLoader(test_data,batch_size=5,shuffle=False)

#%%
class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

#%%
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
#%%
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
inv_normal = NormalizeInverse([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

# a = train_data[100:110]
# images = a[0]

# imgs = []
# for tensor in images:
#     img = unorm(tensor)
#     # img = inv_normal(tensor)
#     img = img.numpy()
#     img = np.transpose(img,(1,2,0))
#     plt.figure()
#     plt.imshow(img)

#     imgs.append(img)
# plt.show()
# images_org = inv_normal(images)
#%%

classes = [
    "male/female",
    'cap',"hairless","short_hair","long_hair_straight","knot",
    "h_white","h_red","h_orange","h_yellow","h_green","h_blue","h_gray","h_purple","h_black",
    "Tshirt_shirt","coat","top","simple/patterned",
    "b_white","b_red","b_orange","b_yellow","b_green","b_blue","b_gray","b_purple","b_black",
    "backpack","bag_hand bag",'no bag',
    "pants","short","skirt",
    "l_white","l_red","l_orange","l_yellow","l_green","l_blue","l_gray","l_purple","l_black",
    'shoes','sandal','hidden',
    "f_white","f_red","f_orange","f_yellow","f_green","f_blue","f_gray","f_purple","f_black",
        ]

writer = SummaryWriter('runs/market-attr')

# to get and show proper images
def show_img(img):
    # unnormalize the images
    img = unorm(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return npimg

# get images 
dataiter = iter(train_loader)
images, labels , id_s = dataiter.next()
# create grid of images
img_grid = torchvision.utils.make_grid(images)
# get and show the unnormalized images
img_grid = show_img(img_grid)
# write to tensorboard
writer.add_image('cifar10 images', img_grid)