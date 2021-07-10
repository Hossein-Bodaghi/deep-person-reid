#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:34:12 2021

@author: hossein
"""

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
from metrics import tensor_metrics, boolian_metrics
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
def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def feature_evaluation(model,test_loader,device):
        
    # taking output from loader 
    torch.cuda.empty_cache()
    model = model.to(device)
    features = []
    model.eval()
    with torch.no_grad():
        
        start = time.time()
        for idx, data in enumerate(test_loader):
            # data = data.to(device) 'list' object has no attribute 'to'
            out_data = model(data[0].to(device))
            features.append(out_data)
            
    finish = time.time()
    print('the time of getting feature is:', finish - start)
    features = torch.cat(features)
    return features

def attr_metrics(attr_net, test_loader, device):
    
    head_metrics = []
    body_metrics = []
    body_type_metrics = []
    leg_metrics = []
    foot_metrics = []
    gender_metrics = []
    bags_metrics = []
    body_color_metrics = []
    leg_color_metrics = []
    foot_color_metrics = []        
    # taking output from loader 
    torch.cuda.empty_cache()
    attr_net = attr_net.to(device)
    model.eval()    
    length = 0
    with torch.no_grad():
        
        start = time.time()
        for idx, data in enumerate(test_loader):
            length += len(data[0])
            # data = data.to(device) 'list' object has no attribute 'to'
            # out_data = attr_net.predict(data[0].to(device))
            out_data = attr_net.predict(data[0].to(device))
            for i in range(len(data)):
                data[i] = data[i].to(device)
            # compute losses and evaluation metrics:
                
            # head     
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            head_metrics.append(metrics)
           
            # body
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            body_metrics.append(metrics)
            
            # body type  
            y = tensor_max(out_data[2])
            metrics = boolian_metrics(data[4].float(),y)
            body_type_metrics.append(metrics)
            
            # leg
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            leg_metrics.append(metrics)
            
            # foot     
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            foot_metrics.append(metrics)
            
            # gender
            y = tensor_max(out_data[5])
            metrics = boolian_metrics(data[7].float(),y)  
            gender_metrics.append(metrics)
            
            # bags
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            bags_metrics.append(metrics)
            
            # body colour 
            y = tensor_max(out_data[7])
            metrics = tensor_metrics(data[9].float(),y)
            body_color_metrics.append(metrics)
            
            # leg colour
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            leg_color_metrics.append(metrics)
            
            # foot colour
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            foot_color_metrics.append(metrics)
            
    finish = time.time()
    print('the time of calculating attributes metrics for {} images is:'.format(length), finish - start)

    return [head_metrics,
    body_metrics,
    body_type_metrics,
    leg_metrics,
    foot_metrics,
    gender_metrics,
    bags_metrics,
    body_color_metrics,
    leg_color_metrics,
    foot_color_metrics]


def map_evaluation(query_featers, gallery_features, dist_matrix):
    
    sorted, indices = torch.sort(dist_matrix)
    # calculating map:
    average_precision = []
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
    return mean_average_precision

#%%
main_path = '/home/hossein/deep-person-reid/my_osnet/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/deep-person-reid/dr_tale/final_attr_org.npy'
path_start = '/home/hossein/deep-person-reid/dr_tale/final_stop.npy'

attr = data_delivery(main_path=main_path,
                     path_attr=path_attr,
                     path_start=path_start,
                     need_collection=True,
                     double=False,
                     need_attr=False)



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
                   num_id=270,
                   feature_dim=512,
                   attr_dim=49,
                   id_inc=True,
                   attr_inc=False)

model1.load_state_dict(feature_net.state_dict())

#%%
'''
Identity based EVALUATION via visual features 
map and rankN
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

query_features = feature_evaluation(model1, query_loader, device=device)
gallery_features = feature_evaluation(model1, gallery_loader, device)

# query_features = query_features.view(-1,512)
# gallery_features = gallery_features.view(-1,512)

start = time.time()
dist_matrix = torch.cdist(query_features, gallery_features)
finish = time.time()
print('the time of distance calculation:' , finish - start)

mean_average_precision = map_evaluation(query_features, gallery_features, dist_matrix)

#%%
query_np = query['id'].to('cpu').numpy()
gallery_np = gallery['id'].to('cpu').numpy()
dist_matrix = dist_matrix.to('cpu').numpy()

rank = torchreid.metrics.rank.evaluate_rank(dist_matrix, query_np, gallery_np, query['cam_id'],
                                            gallery['cam_id'], max_rank=50, use_metric_cuhk03=False, use_cython=False)

#%%

'''
attributes EVALUATION 
precision, recall, accuracy, F1 for attributes
'''

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_idx = [i for i in range(len(attr['id']))]
test_data = MarketLoader3(img_path=main_path,
                          attr=attr,
                          resolution=(256, 128),
                          transform=test_transform,
                          indexes=val_idx) 
batch_size = 200
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

metrics = attr_metrics(model1, test_loader, device)
attr_metrics = torch.zeros((4,46))
part_metrics = torch.zeros((4,10))

for i in range(10):
    for j in range(len(metrics[0])):
        if i==0:
            for k in range(4):
                attr_metrics[k,:5] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==1:
            for k in range(4):
                attr_metrics[k,5:8] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]            
        elif i==2:
            for k in range(4):
                attr_metrics[k,8:9] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k]
        elif i==3:
            for k in range(4):
                attr_metrics[k,9:12] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==4:
            for k in range(4):
                attr_metrics[k,12:15] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==5:
            for k in range(4):
                attr_metrics[k,15:16] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k]
        elif i==6:
            for k in range(4):
                attr_metrics[k,16:19] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==7:
            for k in range(4):
                attr_metrics[k,19:28] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==8:
            for k in range(4):
                attr_metrics[k,28:37] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]
        elif i==9:
            for k in range(4):
                attr_metrics[k,37:] += metrics[i][j][k]
                part_metrics[k,i] += metrics[i][j][k+4]

attr_metrics = attr_metrics/len(metrics[0])
part_metrics = part_metrics/len(metrics[0])           

import numpy as np
a = part_metrics[3].numpy()    

import numpy as np
a = attr_metrics[3].numpy()  
###create excel table

# import xlsxwriter

# attr = ['head','body','body_type','leg','foot','gender','bags','body_c','leg_c','foot_c']
# parts = ['head','body','leg','foot']

# data_attr = []
# data_part = []
