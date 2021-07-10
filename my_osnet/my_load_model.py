#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:56:33 2020

@author: hossein
"""

from torchreid import models , data , optim , engine
import torch
from collections import OrderedDict
import time
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

#%%
def my_load_pretrain(model1 , pretrain_path):
    state_dict = torch.load(pretrain_path)
    model_dict = model1.state_dict()
    new_state_dict = OrderedDict()
    
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items(): # state dict is our loaded weights
            if k.startswith('module.'):
                k = k[7:] # discard module.
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
                
    model_dict.update(new_state_dict)
    model1.load_state_dict(model_dict)   
    
    if len(matched_layers) == 0:
        print(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        ) 
    return model1

#%%
datamanager = data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=10,
    batch_size_test=2,
    transforms=['random_flip', 'random_crop']
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
pretrain_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

model = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)
new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
new_model.to(device)

model2 = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

new_model2 = my_load_pretrain(model2 , pretrain_path = pretrain_path)
new_model2.to(device)
#%%

pretrain_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

model = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)
new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
new_model.to(device)
    
#%%
new_state_dict = new_model.state_dict()
keys = []
for name in new_state_dict:
    keys.append(name)
for idx, m in enumerate(new_model.children()):
    print(idx, '->', m) 

#%%
def feature_model(model):
    new_model1 = torch.nn.Sequential(*list(model.children())[:-2])
    return new_model1

feat_model = feature_model(new_model)
feat_model2 = feature_model(model2)
feat_model1 = torch.nn.Sequential(*list(new_model.children())[:-4])
feat_model.eval()
feat_model2.eval()
feat_model1.eval()
#%%
new_state_dict = feat_model1.state_dict()
keys = []
for name in new_state_dict:
    keys.append(name)
for idx, m in enumerate(new_model.children()):
    print(idx, '->', m) 

#%%
'''
 testing our feature model that does give us 
 feature vectors (stright from convolution network)
'''

test_dataset = datamanager.test_dataset['market1501']['query']  
test_loader = datamanager.test_loader['market1501']['query'] 
for img in test_loader:
    break # img is an dictionary that camera id image and people ids for 100 people (batch size)
start = time.time()
input_images = img['img']
out_put = feat_model(input_images) # pretrained on Market1501
finish = time.time()
print('the time of getting output for batch size of {} is:'.format(input_images.size()),finish - start) # 3 seconds for batch size of 20
print(out_put.size())

start = time.time()
out_put2 = feat_model2(input_images) # pretrained for Image-Net
finish = time.time()
print('the time of getting output for batch size of {} is:'.format(input_images.size()[0]),finish - start) # 3 seconds for batch size of 20
print(out_put2.size())
#%%


print(out_put.size()) # (5,512,1,1)
out_put = torch.Tensor.squeeze(out_put)
print(out_put.size()) # (5,512,1,1) (5,512)
b = out_put[:,4:]
print(b.size()) # (5,508)
pids = img['pid'][:5]
print(pids) #(1255 , 1438 , 319 , 1347 , 96)
#some how eliminating two last layer worked and now we have feature extractor output

#%%
'''
extracting features for training set to be as source data

'''
train_loader = datamanager.train_loader
parser = engine.Engine
def _feature_extraction(data_loader):
    start = time.time()
    f_, pids_ = [], []
    for batch_idx, data1 in enumerate(train_loader):
        imgs, pids = data1['img'] , data1['pid'] 

        features = feat_model(imgs)
        features = torch.Tensor.squeeze(features)
        features = features.data.cpu()
        f_.append(features)
        pids_.extend(pids)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    finish = time.time()
    print('the time of getting output whole training_set:',finish - start) 
    return f_, pids_

f, pids = _feature_extraction(train_loader)
feates = np.asarray(f)
features_save_path = '/home/hossein/anaconda3/envs/torchreid/train_market1501_osnet_ain_x1_0_features.npy'
pids_save_path = '/home/hossein/anaconda3/envs/torchreid/train_market1501_pids.npy'
np.save(features_save_path , feates)
np.save(pids_save_path , pids)
#%%
test_loader = datamanager.test_loader['market1501']['query'] 
source_features = np.load(features_save_path )
for query in test_loader:
    
    start = time.time()
    img , pids = query['img'] , query['pid']
    features = feat_model(img)
    features = torch.Tensor.squeeze(features)
    features = features.data.cpu()
    features = np.asarray(features)
    similarity = cosine_similarity(source_features,features)
    
    
    
    

#%%
'''
 calculating SI for each subset of output features
 
'''

#%%
optimizer = optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

#%%
scheduler = optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

#%%
engine = engine.ImageSoftmaxEngine(
    datamanager,
    new_model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

#%%
engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=True,
    visrank = True
)

#%%
import torch.nn as nn
m = nn.AdaptiveAvgPool2d(7)
input = torch.randn(1, 64, 10, 9)
output = m(input)
print(input.size(),output.size())

#%%
input1 = torch.randn(5,20,1)
input2 = torch.randn(5,20,1)
o = torch.cat((input1,input2),1)