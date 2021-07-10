#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:45:25 2021

@author: hossein
"""

# cd '/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet'
import os
os.chdir('/home/hossein/anaconda3/envs/torchreid/deep-person-reid/my_osnet') 
#%%
import torchreid
"""
version v1 is:
    1) we consider whole vector of output as our target oposite 
    of v1 that we seperatly had a loss function for every collection
"""
from delivery import data_delivery 
from models import my_load_pretrain,MyOsNet,feature_model,MyOsNet2
from loaders import MarketLoader,MarketLoader2,MarketLoader3, Market_folder_Loader
from trainings import train_attr_id , id_onehot,train_collection,train_collection_id
import time
import torch
import torch.nn as nn 
from torchvision import transforms
from torch.utils.data import DataLoader

# import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print('calculation is on:',device)
torch.cuda.empty_cache()

#%%
import numpy as np

main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/deep-person-reid/dr_tale/final_attr_org.npy'
path_start = '/home/hossein/deep-person-reid/dr_tale/final_stop.npy'
path_train = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/bounding_box_train/'

train_names = os.listdir(path_train)
train_names.sort()
    # ids & ids_weights
train_id = []
for name in train_names:
    b = name.split('_')    
    train_id.append(int(b[0]))

start_point = np.load(path_start)    
train_idx = []
test_idx = []
img_names = os.listdir(main_path)
img_names.sort()   
for i,name in enumerate(img_names[:start_point]):
    b = name.split('_')
    if int(b[0]) in train_id:
        train_idx.append(i)
    else:
        test_idx.append(i) 
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

from sklearn.model_selection import train_test_split



train_transform = transforms.Compose([transforms.RandomRotation(degrees=15),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomPerspective(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# random split data to test and train
t_idx, val_idx = train_test_split(train_idx,
                                      test_size=0.25)


torch.save(train_idx, '/home/hossein/deep-person-reid/dr_tale/result/train_idx.pth')
torch.save(test_idx, '/home/hossein/deep-person-reid/dr_tale/result/test_idx.pth')

train_data = MarketLoader3(img_path=main_path,
                          attr=attr,
                          resolution=(256,128),
                          transform=train_transform,
                          indexes=t_idx)
 
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

criterion1 = nn.CrossEntropyLoss()
#criterion2 = nn.MultiLabelSoftMarginLoss(weight=freq_weights)
# criterion2 = nn.BCELoss()
criterion2 = nn.BCEWithLogitsLoss()
params = attr_net.parameters()

optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
#%%

num_epoch = 100
saving_path = '/home/hossein/deep-person-reid/dr_tale/result/'
train_collection_id(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     saving_path,
                     attr['id'].size()[1],
                     device = device,
                     version='V1_4')
#%%
'''
EVALUATION
'''
query_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/query/'
gallery_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'

query = data_delivery(query_path,
                      only_id=True,
                      double = False,
                      need_collection=False,
                      need_attr=True)

gallery = data_delivery(gallery_path,
                        only_id=True,
                        double = False,
                        need_collection=False,
                        need_attr=True)

query_data = Market_folder_Loader(img_path=query_path,
                                  attr=query,
                                  resolution=(256,128))

gallery_data = Market_folder_Loader(img_path=gallery_path,
                                  attr=gallery,
                                  resolution=(256,128))

batch_size = 200
query_loader = DataLoader(query_data,batch_size=batch_size,shuffle=False)
gallery_loader = DataLoader(gallery_data,batch_size=batch_size,shuffle=False)

#%%
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

model_path = '/home/hossein/deep-person-reid/my_osnet/result/attrnet_V1_4.pth'
feature_net = torch.load(model_path)
model1 = MyOsNet2(feat_model,
                   num_id=attr['id'].size()[1],
                   feature_dim=512,
                   attr_dim=49,
                   id_inc=True,
                   attr_inc=False)
model1.load_state_dict(feature_net.state_dict())
# in this part use featers layer of network not the decision or last layer 
# then append this featers tensor 
def evaluation(model,test_loader,device):
        
    # taking output from loader 
    torch.cuda.empty_cache()
    model = model.to(device)
    features = []
    model.eval()
    with torch.no_grad():
        
        start = time.time()
        for idx, data in enumerate(test_loader):
            # data = data.to(device) 'list' object has no attribute 'to'
            out_data = model.get_feature(data[0].to(device))
            features.append(out_data)
            
    finish = time.time()
    print('the time of getting feature is:', finish - start)
    features = torch.cat(features)
    return features

query_featers = evaluation(model1, query_loader, device=device)
gallery_features = evaluation(model1, gallery_loader, device)

start = time.time()
dist_matrix = torch.cdist(query_featers, gallery_features)
finish = time.time()
print('the time of distance calculation:' , finish - start)
sorted, indices = torch.sort(dist_matrix)

# calculating map:
average_precision = []
label_comp = torch.zeros((dist_matrix.size()))
for i in range(len(query_featers)):
    m = 0 # the total positive until that array
    sum_precision = 0
    for j in range(20):
        if query['id'][i] == gallery['id'][indices[i,j]]:
            m += 1
            sum_precision += m/(j+1)
    if m != 0:
        average_precision.append(sum_precision/m)
    else:
        average_precision.append(0)
mean_average_precision = sum(average_precision)/len(average_precision)
# # id comparison
# label_comp = torch.zeros((dist_matrix.size()))
# for i in range(len(query_featers)):
#     for j in range(len(gallery_features)):
#         if query['id'][i] == gallery['id'][j]:
#             label_comp[i,j] = int(query['id'][i])
        