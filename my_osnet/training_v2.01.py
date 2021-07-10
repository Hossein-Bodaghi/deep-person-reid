#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:01:23 2021

@author: hossein

the main difference between this version and v1 is:
    1) we consider whole vector of output as our target oposite 
    of v1 that we seperatly had a loss function for every collection
"""
from delivery import data_delivery 
from models import my_load_pretrain,MyOsNet,feature_model
from loaders import MarketLoader
from trainings import train_attr_id , id_onehot
import torch.nn as nn 
from torchvision import transforms
import torch
# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/deep-person-reid/dr_tale/final_attr_org.npy'
path_start = '/home/hossein/deep-person-reid/dr_tale/final_stop.npy'
# loading attributes

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     path_start=path_start,
                     need_attr=True)
#%%
'''
*
the last piece of data prepation
'''

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.RandomRotation(degrees=15),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomPerspective(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# split data to test and train
train_idx, val_idx = train_test_split(list(range(len(attr['img_names']))),
                                      test_size=0.25)

train_data = MarketLoader(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=transform,
                          indexes=train_idx)
 
test_data = MarketLoader(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=transform,
                          indexes=val_idx) 

train_loader = DataLoader(train_data,batch_size=5,shuffle=True)
test_loader = DataLoader(test_data,batch_size=5,shuffle=True)
#%%
'''
*
load the structure of our network and upload our pretrained weights from downloaded weights 
the output finally is an omni-scale feature extractor network
'''


from torchreid import models    
pretrain_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

model = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=True
)
# new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
feat_model = feature_model(model) # final output (n_batch,512,1,1)
attr_net = MyOsNet(feat_model,
                   num_id=int(attr['id'][-1]+1),
                   feature_dim=512,
                   attr_dim=55,
                   id_inc=True,
                   attr_inc=True)

#%%
'''
*
criterion1 is categorical cross entropy and will be used for:
    head,body,leg,foot,colours
criterion2 is binary cross entropy:
    gender,body_typehead_metrics['collection']['recall'].append(np.mean(rt0))
criterion3 is multi label soft margin loss:
    attr['attributes'] whole attr vector as target
    nn.MULTILABELSOFTMARGINLOSS()
    weights contains a 55 sized vector from 0 to 5 that is has opposit correlation with
    frequency 
'''

freq_weights = attr['freq_weights']
id_weights = attr['id_weights']

lr = 0.0001

criterion1 = nn.CrossEntropyLoss(weight=id_weights)
criterion2 = nn.MultiLabelSoftMarginLoss(weight=freq_weights)
# criterion2 = nn.BCELoss()
# 
params = attr_net.parameters()

optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)

#%%
num_epoch = 3
saving_path = '/home/hossein/anaconda3/envs/torchreid/my_osnet/'
train_loss,test_loss,F1_train,F1_test,Acc_train,Acc_test=train_attr_id(attr_net,
                  train_loader,
                  test_loader,
                  num_epoch,
                  optimizer,
                  scheduler,
                  criterion1,
                  criterion2,
                  saving_path,
                  attr['id'][-1]+1,
                  id_inc=True, version='v1.02')
#%%
# o = attr_net(data[0])
# for idx, data in enumerate(train_loader):
    
#     # forward step
#     optimizer.zero_grad()
#     out_data = attr_net(data[0])