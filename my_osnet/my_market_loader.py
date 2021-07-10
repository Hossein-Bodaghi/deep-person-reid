#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:18:18 2021

@author: hossein

our data loader in pytorch 
it should be like this:
    when get_item called it takes the address of 
    an image and take it as a picture 
    
"""

from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image
import numpy as np
import os

def get_image(addr,height,width):
    test_image = Image.open(addr)
    ratio_w = width / test_image.width
    ratio_h = height / test_image.height
    if ratio_w < ratio_h:
      # It must be fixed by width
      resize_width = width
      resize_height = round(ratio_w * test_image.height)
    else:
      # Fixed by height
      resize_width = round(ratio_h * test_image.width)
      resize_height = height
    image_resize = test_image.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGB', (width, height), (255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    # img = np.array(background)
    # img = img.astype(np.float32)
    # print('Size changed form (Hossein):', [test_image.width, test_image.height], 'to:', img.shape )   
    return background



class MarketLoader(Dataset):
    '''
    attr: an attribute vector (zeros eliminated).
    img_path: the folder of our source images. 
    resolution: the final dimentions of images (height,width)
    transform: images transformations
    
    '''
    def __init__(self,img_path,attr,resolution,transform=None):

        self.attr = attr
        self.img_path = img_path
        img_names = os.listdir(img_path)
        self.img_names = img_names[:len(attr)]        
        self.resolution = resolution
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.attr)
    
    def __getitem__(self,idx):
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
        sample = self.transform(img)
        
        return (sample , self.attr[idx])

# #%%
# import torch

# main_path = '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
# path_attr = '/home/hossein/market1501_label/final_attr.npy'
# path_start = '/home/hossein/market1501_label/final_stop.npy'

# # loading attributes
# start_point = np.load(path_start)
# attr_vec_np = np.load(path_attr)
# attr_vec_np = attr_vec_np.astype(np.float32)
# attr_vec_torch = torch.from_numpy(attr_vec_np)
# attr_vec_torch = attr_vec_torch[:start_point]

# data = MarketLoader(main_path,attr_vec_torch,(256,128))

# #%%

# a,b = data[-1]
    
    
#%%

# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample