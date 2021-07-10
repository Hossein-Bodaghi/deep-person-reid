#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:00:10 2020

@author: hossein
"""
#%%
from torchreid import models , data
from torch import load
from collections import OrderedDict

#%%
datamanager = data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)
#%%
# make a osnet model 
pretrain_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
osnet = models.osnet_ain.osnet_ain_x1_0(pretrained=False) # should be False becuase true load imagenet pretrained and architects are different
# osnet_expriment = models.osnet_ain.init_pretrained_weights(osnet , 'osnet_ain_x1_0_msmt17_256x128')
# the above is for loading pretrained for imagenet 

model = models.build_model(
    name='osnet_ain_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)

dictt = model.state_dict()
for idx, m in enumerate(model.modules()):
    print(idx, '->', m)
    
# the above is for loading pretrained from build model


#%%
def my_load_pretrain(model1 , pretrain_path):
    state_dict = load(pretrain_path)
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
new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
new_state_dict = new_model.state_dict()
for idx, m in enumerate(new_model.modules()):
    print(idx, '->', m)
i = 0
for name in new_state_dict:
    i += 1
    print(i , '->' , name)
a = new_state_dict['classifier.weight']
b = model.state_dict()['classifier.weight']
#%%






#%%

# load weights from downloaded pretrained model and assigne to osnet 
model_state_dict = load('/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
osnet.load_state_dict(model_state_dict)

#%%
state_dict = mymodel.state_dict()
i = 0
for name in model_state_dict:
    if name == 'classifier.bias111':
        name = 'lets_change'
    i += 1
    print(i , '->' , name)
a = state_dict['classifier.bias']
print(a)
#%%
for state in model_state_dict:
    print(state)

fc_1_bias = model_state_dict['fc.1.bias']
a = model_state_dict['conv4.1.conv2.3.layers.1.bn.running_var']
import torch
new_fc_weight = torch.zeros((512,512))
model_state_dict['fc.1.weight'] = new_fc_weight

for i in a:
    print(i)
print(a)
print(model_state_dict['fc.1.weight'] )

#%%
# load weights from downloaded pretrained model and assigne to osnet by torchreid's function.
from torchreid.utils import torchtools
weight_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
model_state_dict = load(weight_path)
new = load(weight_path)
os_pretrain = osnet.state_dict()
for key , value in list(model_state_dict.items()):
    lis = key.split('.')
    a = ''
    if lis[0] == 'module':
        for i in range(1,len(lis)):
            if i == 1:
                a = lis[i]
            else:
                a = a + '.' + lis[i]
    del model_state_dict[key]
    model_state_dict[a] = value

import torch
torch.save(model_state_dict, '/home/hossein/anaconda3/envs/torchreid/my_osnet.pth')
i=0
for name in os_pretrain:
    print(name)
    if os_pretrain[name][0] ==  model_state_dict[name][0]:
        i += 1




new_dict = {}
for name in model_state_dict:
    lis = name.split('.')
    a = ''
    if lis[0] == 'module':
        for i in range(1,len(lis)):
            if i == 1:
                a = lis[i]
            else:
                a = a + '.' + lis[i]
    new_dict[a] = model_state_dict[name]

i = 0            
for name in state_dict:
    if name == 'classifier.bias1111':
        name = 'lets_change'
    i += 1
    print(i , '->' , name)
        
        
torchtools.load_pretrained_weights(osnet, new_dict)

#%%
# from collections import OrderedDict
from torchreid.utils import torchtools
weight_path = '/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
model_state_dict = load(weight_path)
my_state_dict = load('/home/hossein/anaconda3/envs/torchreid/my_osnet.pth')
# new_dict = {}
# new_dict = OrderedDict(new_dict)
for key , value in list(model_state_dict.items()):
    lis = key.split('.')
    a = ''
    if lis[0] == 'module':
        for i in range(1,len(lis)):
            if i == 1:
                a = lis[i]
            else:
                a = a + '.' + lis[i]
    del model_state_dict[key]
    model_state_dict[a] = value

model = torchtools.load_pretrained_weights(osnet, '/home/hossein/anaconda3/envs/torchreid/my_osnet.pth')
model.state_dict()
#%%
# the osnet has two blocks for classification
# firs is a Linear and BatchNormalization and relue 512 in and 512 out
# second one is just a Linear 512 in and 1000 out
from torch import nn
my_osnet = nn.Sequential(*list(osnet.modules())[:-2]) # remove just last layer of osnet

#%%
model = torchtools.load_checkpoint(weight_path)
for idx, m in enumerate(osnet.modules()):
    print(idx, '->', m)
for idx, m in enumerate(osnet.children()):
    print(idx, '->', m) 
for idx, m in enumerate(osnet.state_dict()):
    print(idx, '->', m)

from torch import nn
my_osnet = nn.Sequential(*list(osnet.children())[:-1])
for idx, m in enumerate(my_osnet.children()):
    print(idx, '->', m) 
    
for idx, m in enumerate(my_osnet.modules()):
    print(idx, '->', m)    

#%%
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
osnet.apply(init_weights)
#%%
# making our desire network not ready yet
from torch import nn
class Si_osnet(nn.Module):
    def __init__(self , main_model):
        super(Si_osnet , self).__init__()
        self.feature_extractor = nn.Sequential(*list(main_model.children())[:-1])
    def forward(self, x):
        out = self.feature_extractor(x)
        return out
#%%
si_net =  Si_osnet(osnet)   
#%%
# loading my photos dataset
from torchreid import data
datamanager = data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)
number_of_person = datamanager.num_train_pids
test_dataset = datamanager.test_dataset['market1501']['gallery']  
test_loader = datamanager.test_loader['market1501']['gallery'] 
for img in test_loader:
    break

#%%
import cv2 
path = test_dataset[0][0]
image = cv2.imread(path)

#%%
import torch.tensor as ts
from torchvision.transforms import transforms
image1 = ts(image)
trans = transforms.ToTensor()
image2 = trans(image)
out = my_osnet(image2.unsqueeze(0))
out = my_osnet(img['img'])
out2 = osnet(img['img'][:2])
my_osnet.eval()
out = my_osnet.forward(img['img'][:1])
out = si_net(img['img'][:2])
out3 = osnet.featuremaps(img['img'][:2])

#%%
modulelist = list(osnet.features.modules())
a = osnet.layer()
#%%
from torchreid import data
VIPeR = data.datasets.image.viper.VIPeR()
    
#%%
from torchreid.utils import FeatureExtractor