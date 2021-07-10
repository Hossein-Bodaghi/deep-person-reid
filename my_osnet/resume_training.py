#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:07:23 2021

@author: hossein
version v1 is:
    1) we consider whole vector of output as our target oposite 
    of v1 that we seperatly had a loss function for every collection
"""
import os
os.chdir('/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet')
 
#%%
from delivery import data_delivery 
from models import feature_model,MyOsNet2
from loaders import MarketLoader3
from trainings import train_collection_id
import torch
import torch.nn as nn 
from torchvision import transforms
from torch.utils.data import DataLoader
# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)

#%%

main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/deep-person-reid/dr_tale/final_attr_org.npy'
path_start = '/home/hossein/deep-person-reid/dr_tale/final_stop.npy'
# loading attributes

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     path_start=path_start,
                     need_collection=True,
                     double=False,
                     need_attr=False)

#%%
for key , value in attr.items():
  try: print(key , 'size of {} tensor is: \t {} \n'.format(key,(value.size())))
  except TypeError:
    print(key,'\n')
    
#%%
    
'''
*
the last piece of data prepation
'''

train_idx_path = '/home/hossein/deep-person-reid/dr_tale/result/train_idx.pth'
test_idx_path = '/home/hossein/deep-person-reid/dr_tale/result/test_idx.pth'

train_idx = torch.load(train_idx_path,map_location= torch.device(device)) 
val_idx = torch.load(test_idx_path,map_location= torch.device(device))


train_transform = transforms.Compose([transforms.RandomRotation(degrees=15),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomPerspective(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    

train_data = MarketLoader3(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=train_transform,
                          indexes=train_idx)
 
test_data = MarketLoader3(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          transform=test_transform,
                          indexes=val_idx) 

batch_size = 40
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

#%%
torch.cuda.empty_cache()
'''
*
load the structure of our network and upload our pretrained weights from downloaded weights 
the output finally is an omni-scale feature extractor network
'''

from torchreid import models    
#pretrain_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

model = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=True
)
# new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
feat_model = feature_model(model) # final output (n_batch,512,1,1)
attr_net = MyOsNet2(feat_model,
                   num_id=attr['id'].size()[1],
                   feature_dim=512,
                   attr_dim=49,
                   id_inc=True,
                   attr_inc=False)

model_path = '/home/hossein/deep-person-reid/dr_tale/result/attrnet_V1_4_epoch29.pth'
feature_net = torch.load(model_path, map_location=device)
attr_net.load_state_dict(feature_net.state_dict())
attr_net = attr_net.to(device)

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
freq_weights = freq_weights
id_weights = id_weights

lr = 0.0001
stop_epoch = 29

criterion1 = nn.CrossEntropyLoss()
#criterion2 = nn.MultiLabelSoftMarginLoss(weight=freq_weights
# criterion2 = nn.BCELoss()
criterion2 = nn.BCEWithLogitsLoss()
params = attr_net.parameters()

opt_state_dict_path = '/home/hossein/deep-person-reid/dr_tale/result/optimizer_V1_4_epoch29.pth'
opt_state_dict_path = torch.load(opt_state_dict_path)
optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

for i in range(0,stop_epoch):
    scheduler.step()
    
optimizer.load_state_dict(opt_state_dict_path)
#%%

stop_epoch = 29
num_epoch = 50
saving_path = '/home/hossein/deep-person-reid/dr_tale/result/'

train_loss_path = '/home/hossein/deep-person-reid/dr_tale/result/trainloss_V1_4.pth'
test_loss_path = '/home/hossein/deep-person-reid/dr_tale/result/testloss_V1_4.pth'
train_F1_path = '/home/hossein/deep-person-reid/dr_tale/result/trainF1_V1_4.pth'
test_F1_path = '/home/hossein/deep-person-reid/dr_tale/result/testF1_V1_4.pth'

loss_train = torch.load(train_loss_path)
loss_test = torch.load(test_loss_path)
train_F1 = torch.load(train_F1_path)
test_F1 = torch.load(test_F1_path)
train_collection_id(num_epoch=num_epoch,
                     attr_net=attr_net,
                     train_loader=train_loader,
                     test_loader=test_loader,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     criterion1=criterion1,
                     criterion2=criterion2,
                     saving_path=saving_path,
                     num_id=attr['id'].size()[1],
                     device=device,
                     version='V1_5',
                     resume=True,
                     loss_train=loss_train,
                     loss_test=loss_test,
                     train_F1=train_F1,
                     test_F1=test_F1,
                     stop_epoch=stop_epoch)