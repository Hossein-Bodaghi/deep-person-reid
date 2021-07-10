#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:36:35 2021

@author: hossein

this is second version of training on market 1501 attribute dettection
"""



import torch
import time
import numpy as np
import os
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%%
'''
*
a function to take an image from path and change its size to a new height and width 
it is different from library functions because consider the proportion of h/w of base image
and if the proportion of new h/w is different it will add a white background
'''

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
        return background            
   


#%%
'''
*
we need this vectors seperately from our attribute vector 

1) head (cap,bald,sh,lhs,lhn)
2) body (shirt,coat,top) 
3) body type (simple,patterned)
4) leg (pants,shorts,skirt)
5) foot (shoes,sandal,hidden)
6) gender (male,female)sor(),
7) bags (backpack,hand bag,nothing)
8) body colour (9 colours)
9) leg colour (9 colours)
10) foot colour (9 colours)


attr = [
    "male/female0",
    'cap1',"hairless2","short_hair3","long_hair_straight4","knot5",
    "h_white6","h_red7","h_orange8","h_yellow9","h_green10","h_blue11","h_gray12","h_purple13","h_black14",
    "Tshirt_shirt15","coat16","top17","simple/patterned18",
    "b_white19","b_red20","b_orange21","b_yellow22","b_green23","b_blue24","b_gray25","b_purple26","b_black27",
    "backpack28","bag_hand bag29",'no bag30',
    "pants31","short32","skirt33",
    "l_white34","l_red35","l_orange36","l_yellow37","l_green38","l_blue39","l_gray40","l_purple41","l_black42",
    'shoes43','sandal44','hidden45',
    "f_white46","f_red47","f_orange48","f_yellow49","f_green50","f_blue51","f_gray52","f_purple53","f_black54",
        ]
'''
main_path = '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
path_attr = '/home/hossein/market1501_label/final_attr.npy'
path_start = '/home/hossein/market1501_label/final_stop.npy'
# loading attributes
start_point = np.load(path_start)
attr_vec_np = np.load(path_attr)# loading attributes
attr_vec_np = attr_vec_np.astype(np.int32)
attr_vec_np = attr_vec_np[:start_point]
attr_vec_double = np.append(attr_vec_np,attr_vec_np,axis=0)

img_names = os.listdir(main_path)
img_names.sort()
img_names = img_names[:start_point]
img_names = np.array(img_names)
img_names_double = list(np.append(img_names,img_names,axis=0))
id_ = []
for name in img_names_double:
    b = name.split('_')
    id_.append(int(b[0]))

head = []
body = []
body_type = []
leg = []
foot = []
gender = []
bags = []
body_colour = []
leg_colour = []
foot_colour = []

for vec in attr_vec_double:
    
    gender.append(vec[0])
    head.append(vec[1:6])
    body.append(vec[15:18])
    body_type.append(vec[18])
    leg.append(vec[31:34])
    foot.append(vec[43:46])
    bags.append(vec[28:31])
    body_colour.append(vec[19:28])
    leg_colour.append(vec[34:43])
    foot_colour.append(vec[46:])
    
# one hot id vectors
last_id = id_[-1]
id1 = torch.zeros((len(id_),last_id))
for i in range(len(id1)):
    a = id_[i]
    id1[i,a-1] = 1
    
    
attr = {'id':id1,
        'img_names':np.array(img_names_double),
        'head':torch.from_numpy(np.array(head)),
        'body':torch.from_numpy(np.array(body)),
        'body_type':torch.tensor(body_type),
        'leg':torch.from_numpy(np.array(leg)),
        'foot':torch.from_numpy(np.array(foot)),
        'gender':torch.tensor(gender),
        'bags':torch.from_numpy(np.array(bags)),
        'body_colour':torch.from_numpy(np.array(body_colour)),
        'leg_colour':torch.from_numpy(np.array(leg_colour)),
        'foot_colour':torch.from_numpy(np.array(foot_colour))}

#%%
'''
*

market costume data loader
'''    
from torch.utils.data import Dataset 
from torchvision import transforms
    
class MarketLoader(Dataset):
    '''
    attr is a dictionary contains:
        
        1) head (cap,bald,sh,lhs,lhn)
        2) body (shirt,coat,top) 
        3) body_type (simple,patterned)
        4) leg (pants,shorts,skirt)
        5) foot (shoes,sandal,hidden)
        6) gender (male,female)
        7) bags (backpack,hand bag,nothing)
        8) body_colour (9 colours)
        9) leg_colour (9 colours)
        10) foot_colour (9 colours)
        11) img_names: names of images in source path
        12) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,attr,resolution,transform,indexes):

         
        self.img_path = img_path
        
        self.id = attr['id'][indexes]
        self.img_names = attr['img_names'][indexes]
        self.head = attr['head'][indexes]
        self.body = attr['body'][indexes]
        self.body_type = attr['body_type'][indexes]
        self.leg = attr['leg'][indexes]
        self.foot = attr['foot'][indexes]
        self.gender = attr['gender'][indexes]
        self.bags = attr['bags'][indexes]
        self.body_colour = attr['body_colour'][indexes]
        self.leg_colour = attr['leg_colour'][indexes]
        self.foot_colour = attr['foot_colour'][indexes]
        
        self.resolution = resolution
        
        self.transform_complex = transform            

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # imagenet normalization
        
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self,idx):
        
        img = get_image(self.img_path+self.img_names[idx], self.resolution[0], self.resolution[1])
       
        # we have doublicated the images path so in one epoch half of images will be 
        # augmented and the other half wont change        
        
        t = torch.empty(1).random_(2)
        if t == 0:
            self.transform = self.transform_complex
        else:
            self.transform = self.transform_simple
        img = self.transform(img)
        return (img,
                self.id[idx].to(device),
                self.head[idx].to(device),
                self.body[idx].to(device),
                self.body_type[idx].to(device),
                self.leg[idx].to(device),
                self.foot[idx].to(device),
                self.gender[idx].to(device),
                self.bags[idx].to(device),
                self.body_colour[idx].to(device),
                self.leg_colour[idx].to(device),
                self.foot_colour[idx].to(device))
#%%

'''
if you dont want to use dataloader from pytorch
market costume data loader
'''    
from torch.utils.data import Dataset 
from torchvision import transforms
    
class MarketLoader2(Dataset):
    '''
    attr is a dictionary contains:
        
        1) head (cap,bald,sh,lhs,lhn)
        2) body (shirt,coat,top) 
        3) body_type (simple,patterned)
        4) leg (pants,shorts,skirt)
        5) foot (shoes,sandal,hidden)
        6) gender (male,female)
        7) bags (backpack,hand bag,nothing)
        8) body_colour (9 colours)
        9) leg_colour (9 colours)
        10) foot_colour (9 colours)
        11) img_names: names of images in source path
        12) id is the identification number of each picture
    img_path: the folder of our source images. '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
    resolution: the final dimentions of images (height,width) (256,128)
    transform: images transformations
    
    
    '''
    def __init__(self,img_path,attr,resolution,transform,indexes):

         
        self.img_path = img_path
        
        self.id = attr['id'][indexes]
        self.img_names = attr['img_names'][indexes]
        self.head = attr['head'][indexes]
        self.body = attr['body'][indexes]
        self.body_type = attr['body_type'][indexes]
        self.leg = attr['leg'][indexes]
        self.foot = attr['foot'][indexes]
        self.gender = attr['gender'][indexes]
        self.bags = attr['bags'][indexes]
        self.body_colour = attr['body_colour'][indexes]
        self.leg_colour = attr['leg_colour'][indexes]
        self.foot_colour = attr['foot_colour'][indexes]
        
        self.resolution = resolution
        
        self.transform_complex = transform            

        self.transform_simple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # imagenet normalization
        
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self,idx):
        
        imgs = torch.empty((len(self.head[idx]),3,self.resolution[0],self.resolution[1]),device=device)
        if self.img_names[idx].ndim == 0:
            a = np.expand_dims(self.img_names[idx], axis=0)
            for i,name in enumerate(a):
                img = get_image(self.img_path+name, self.resolution[0], self.resolution[1])
                # img = get_image(os.path.join(self.img_path,self.img_names[idx]), self.resolution[0], self.resolution[1])
                t = torch.empty(1).random_(2)
            
                # we have doublicated the images path so in one epoch half of images will be 
                # augmented and the other half wont change 
                if t == 0:
                    self.transform = self.transform_complex
                else:
                    self.transform = self.transform_simple
                
                imgs[i] = self.transform(img)
        else:
            for i,name in enumerate(self.img_names[idx]):
                img = get_image(self.img_path+name, self.resolution[0], self.resolution[1])
                # img = get_image(os.path.join(self.img_path,self.img_names[idx]), self.resolution[0], self.resolution[1])
                t = torch.empty(1).random_(2)
            
                # we have doublicated the images path so in one epoch half of images will be 
                # augmented and the other half wont change 
                if t == 0:
                    self.transform = self.transform_complex
                else:
                    self.transform = self.transform_simple
                
                imgs[i] = self.transform(img)
        
        return (imgs,
                self.id[idx].to(device),
                self.head[idx].to(device),
                self.body[idx].to(device),
                self.body_type[idx].to(device),
                self.leg[idx].to(device),
                self.foot[idx].to(device),
                self.gender[idx].to(device),
                self.bags[idx].to(device),
                self.body_colour[idx].to(device),
                self.leg_colour[idx].to(device),
                self.foot_colour[idx].to(device))

#%%
'''
*

when load a pretrained model from torchreid it just brings imagenet trained models
so if we want to bring pretrained on other datasets we should use this function

'''

from collections import OrderedDict

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
    pretrained=False
)
new_model = my_load_pretrain(model , pretrain_path = pretrain_path)
new_model.to(device)

def feature_model(model):
    new_model1 = torch.nn.Sequential(*list(model.children())[:-2])
    return new_model1

feat_model = feature_model(new_model) # final output (n_batch,512,1,1)


#%%
'''
* 
this is our network in this version it just take output from features of
original omni-scale network and for each attribute has a seperate linear 
layer for classification

'''

import torch.nn as nn
        
class MyOsNet(nn.Module):
    
    def __init__(self,model):
        
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batchnormalization = nn.BatchNorm1d(num_features=512)
        self.dropout = nn.Dropout(0.3)


        self.model = model
        self.linear = nn.Linear(in_features=512 , out_features=512)
        self.id_lin = nn.Linear(in_features=558 , out_features=261)
        self.head_lin = nn.Linear(in_features=512 , out_features=5)
        self.body_lin = nn.Linear(in_features=512 , out_features=3)
        self.body_type_lin = nn.Linear(in_features=512 , out_features=1)
        self.leg_lin = nn.Linear(in_features=512 , out_features=3)
        self.foot_lin = nn.Linear(in_features=512 , out_features=3)
        self.gender_lin = nn.Linear(in_features=512 , out_features=1)
        self.bags_lin = nn.Linear(in_features=512 , out_features=3)
        self.body_colour_lin = nn.Linear(in_features=512 , out_features=9)
        self.leg_colour_lin = nn.Linear(in_features=512 , out_features=9)
        self.foot_colour_lin = nn.Linear(in_features=512 , out_features=9)       
        

        
    def forward(self, x):
        
        features = self.model(x)
        features = features.view(-1,512)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        out_head = self.softmax(self.head_lin(features))
        out_body = self.softmax(self.body_lin(features))
        out_body_type = self.sigmoid(self.body_type_lin(features))
        out_leg = self.softmax(self.leg_lin(features))
        out_foot = self.softmax(self.foot_lin(features))
        out_gender = self.sigmoid(self.gender_lin(features))
        out_bags = self.softmax(self.bags_lin(features))
        out_body_colour = self.softmax(self.body_colour_lin(features))
        out_leg_colour = self.softmax(self.leg_colour_lin(features))
        out_foot_colour = self.softmax(self.foot_colour_lin(features))

        # id will be added 
        return (out_head,
                out_body,
                out_body_type,
                out_leg,
                out_foot,
                out_gender,
                out_bags,
                out_body_colour,
                out_leg_colour,
                out_foot_colour)
    
#%%

'''
*
this part is for calculating accuracy & F1 & precision & recall:
    we have three types of each of above mentioned:
        1) specific indcies for every tag for example recall for hand_bags or hat
        2) specific indices for every attribute (collection) for example for body or leg
every key in collection is a percentage
every key in tag is a vector a percantage for each tag in a collection 
for example for head precision is a (1,5) tensor

precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = tp/(tp+tn+fp+fn)
f1 = 2*(precision+recall)/(precision+recall)

    '''

gender_metrics = {'collection':{'loss':[],
                              'precision':[],
                              'accuracy':[],
                              'recall':[],
                              'F1':[]},
                'tag':{'tags':['male','female']}}

head_metrics = {'collection':{'loss':[],
                              'precision':[],
                              'accuracy':[],
                              'recall':[],
                              'F1':[]},
                'tag':{'tags':['cap',
                               'bald',
                               'short hair',
                               'long hair straight',
                               'long hair knot'],
                       'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

body_metrics = {'collection':{'loss':[],
                              'precision':[],
                              'accuracy':[],
                              'recall':[],
                              'F1':[]},
                'tag':{'tags':['shirt',
                               'coat',
                               'top'],
                       'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

body_type_metrics = {'collection':{'loss':[],
                                   'precision':[],
                                   'accuracy':[],
                                   'recall':[],
                                   'F1':[]},
                'tag':{'tags':['simple',
                               'patterned']}}

leg_metrics = {'collection':{'loss':[],
                             'precision':[],
                             'accuracy':[],
                             'recall':[],
                             'F1':[]},
                'tag':{'tags':['pants',
                               'shorts',
                               'skirt'],
                       'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

foot_metrics = {'collection':{'loss':[],
                              'precision':[],
                              'accuracy':[],
                              'recall':[],
                              'F1':[]},
                'tag':{'tags':['shoes',
                               'sandal',
                               'hidden'],
                       'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

bags_metrics = {'collection':{'loss':[],
                              'precision':[],
                              'accuracy':[],
                              'recall':[],
                              'F1':[]},
                'tag':{'tags':['backpack',
                               'bags',
                               'no bags'],
                       'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

body_colour_metrics = {'collection':{'loss':[],
                                     'precision':[],
                                     'accuracy':[],
                                     'recall':[],
                                     'F1':[]},
                'tag':{'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

leg_colour_metrics = {'collection':{'loss':[],
                                    'precision':[],
                                    'accuracy':[],
                                    'recall':[],
                                    'F1':[]},
                'tag':{'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

foot_colour_metrics = {'collection':{'loss':[],
                                    'precision':[],
                                    'accuracy':[],
                                    'recall':[],
                                    'F1':[]},
                'tag':{'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

id_metrics = {'collection':{'loss':[],
                            'precision':[],
                            'accuracy':[],
                            'recall':[],
                            'F1':[]},
                'tag':{'precision':[],
                       'accuracy':[],
                       'recall':[],
                       'F1':[]}}

#%%

'''
*

first we define tp tn fn fp and then manipulate them to 
calculate precision recall and accuracy and f1 

precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = tp/(tp+tn+fp+fn)
f1 = 2*(precision+recall)/(precision+recall)
'''
# predict = torch.zeros((100,3))
# target = torch.zeros((100,3))
# for i in range(100):
#     t = int(torch.empty(1).random_(3))
#     y = int(torch.empty(1).random_(3))
#     predict[i,t] = 1
#     target[i,y] = 1

def tensor_metrics(target,predict):
    eps = 2e-16
    
    true_positive = torch.zeros((predict.size()[1]))
    true_negative = torch.zeros((predict.size()[1]))
    false_positive = torch.zeros((predict.size()[1]))
    false_negative = torch.zeros((predict.size()[1]))
    
    true_positive_total = 0
    true_negative_total = 0
    false_positive_total = 0
    false_negative_total = 0
    
    
    for i in range(len(predict)):
        for j in range(predict.size()[1]):
            if predict[i,j] == target[i,j] and target[i,j] == 1:
                true_positive[j] += 1
                true_positive_total += 1
            elif predict[i,j] == target[i,j] and target[i,j] == 0:
                true_negative[j] += 1
                true_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 1:
                false_negative[j] += 1
                false_negative_total += 1
            elif predict[i,j] != target[i,j] and target[i,j] == 0:
                false_positive[j] += 1
                false_positive_total += 1
    
    precision = torch.zeros((predict.size()[1]))
    recall = torch.zeros((predict.size()[1]))
    accuracy = torch.zeros((predict.size()[1]))
    f1 = torch.zeros((predict.size()[1]))
    
    precision_total = true_positive_total/(true_positive_total+false_positive_total+eps)
    recall_total = true_positive_total/(true_positive_total+false_negative_total+eps)
    accuracy_total = true_positive_total/(true_positive_total+false_negative_total+true_negative_total+false_positive_total+eps)
    f1_total = 2*(precision_total*recall_total)/(precision_total+recall_total+eps)
    
    for j in range(predict.size()[1]):
        precision[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps)
        recall[j] = true_positive[j]/(true_positive[j]+false_negative[j]+eps)
        accuracy[j] = true_positive[j]/(true_positive[j]+false_negative[j]+true_negative[j]+false_positive[j]+eps)
        f1[j] = 2*(precision[j]*recall[j])/(precision[j]+recall[j]+eps)       
        
    return [precision,
            recall,
            accuracy,
            f1,
            precision_total,
            recall_total,
            accuracy_total,
            f1_total]


def boolian_metrics(target,predict):
    eps = 2e-16
    true_positive_total = 0
    true_negative_total = 0
    false_positive_total = 0
    false_negative_total = 0
    
    
    for i in range(len(predict)):
        
        if predict[i] == target[i] and target[i] == 1:
            true_positive_total += 1
        elif predict[i] == target[i] and target[i] == 0:
            true_negative_total += 1
        elif predict[i] != target[i] and target[i] == 1:
            false_negative_total += 1
        elif predict[i] != target[i] and target[i] == 0:
            false_positive_total += 1
            
    precision_total = true_positive_total/(true_positive_total+false_positive_total+eps)
    recall_total = true_positive_total/(true_positive_total+false_negative_total+eps)
    accuracy_total = true_positive_total/(true_positive_total+false_negative_total+true_negative_total+false_positive_total+eps)
    f1_total = 2*(precision_total*recall_total)/(precision_total+recall_total+eps)
    
    return [precision_total,
            recall_total,
            accuracy_total,
            f1_total]

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
train_idx, val_idx = train_test_split(list(range(len(img_names_double))),
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
criterion1 is categorical cross entropy and will be used for:
    head,body,leg,foot,colours
criterion2 is binary cross entropy:
    gender,body_typehead_metrics['collection']['recall'].append(np.mean(rt0))
'''
lr = 0.001
attr_net = MyOsNet(feat_model).to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()

params = attr_net.parameters()

optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)

#%%
'''
*
functions which are needed for training proccess.

tensor_max: take a matrix and return a matrix with one hot vectors
lis2tensor: take a list cintaining torch tensors and return a torch matrix  

'''
def tensor_max(tensor):
    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size()).scatter_(1, idx, 1.)
    return y

def list2tensor(list1):
    tensor = torch.zeros((len(list1),list1[0].size()[0]))
    for i in range(len(list1)):
        tensor[i] = list1[i]
    return tensor    
    
#%%
num_epoch = 2
# def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
train_loss = []
test_loss = []
F1_train = []
F1_test = []

for epoch in range(1,num_epoch+1):
    
    attr_net.train()
    loss_e = []
    loss_t = []
    ft_train = []
    ft_test = []
    
    for idx, data in enumerate(train_loader):
        
        # forward step
        optimizer.zero_grad()
        out_data = attr_net(data[0])
        
        # compute losses and evaluation metrics:
            
        # head 
        loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
        y = tensor_max(out_data[0])
        metrics = tensor_metrics(data[2].float(),y)
        ft_train.append(metrics[7])
        
        # body
        loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
        y = tensor_max(out_data[1])
        metrics = tensor_metrics(data[3].float(),y)
        ft_train.append(metrics[7])
        
        # body type
        loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
        y = tensor_max(out_data[2])
        metrics = boolian_metrics(data[4].float(),y)
        ft_train.append(metrics[3])
        
        # leg
        loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
        y = tensor_max(out_data[3])
        metrics = tensor_metrics(data[5].float(),y)
        ft_train.append(metrics[7])
        
        # foot 
        loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
        y = tensor_max(out_data[4])
        metrics = tensor_metrics(data[6].float(),y)  
        ft_train.append(metrics[7])
        
        # gender
        loss5 = criterion2(out_data[5].squeeze(),data[7].float())
        y = tensor_max(out_data[5])
        metrics = boolian_metrics(data[7].float(),y)  
        ft_train.append(metrics[3])
        
        # bags
        loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
        y = tensor_max(out_data[6])
        metrics = tensor_metrics(data[8].float(),y)
        ft_train.append(metrics[7])
        
        # body colour
        loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
        y = tensor_max(out_data[7])
        metrics = tensor_metrics(data[9].float(),y)
        ft_train.append(metrics[7])
        
        # leg colour
        loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
        y = tensor_max(out_data[8])
        metrics = tensor_metrics(data[10].float(),y)
        ft_train.append(metrics[7])
        
        # foot colour
        loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
        y = tensor_max(out_data[9])
        metrics = tensor_metrics(data[11].float(),y)      
        ft_train.append(metrics[7])
        
        # total loss
        loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
        loss_e.append(loss.item())
        
        # backward step
        loss.backward()
        
        # optimization step
        optimizer.step()
        scheduler.step()
        # print log
        if idx % 1 == 0:
            print('Train Epoch: {} [{}/{} , idx {}, lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                epoch, idx * len(data[8]), len(train_loader), idx, optimizer.param_groups[0]['lr'], loss.item(),np.mean(ft_train)))
   
    train_loss.append(np.mean(loss_e))
    F1_train.append(np.mean(ft_train))
    torch.save(train_loss,'/home/hossein/anaconda3/envs/torchreid/my_osnet/trainloss_v1_02.pth')
    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            
            out_data = attr_net(data[0])
            
            # compute losses and evaluation metrics:
                
            # head 
            loss0 = criterion1(out_data[0],data[2].argmax(dim=1))        
            y = tensor_max(out_data[0])
            metrics = tensor_metrics(data[2].float(),y)
            ft_test.append(metrics[7])
            
            # body
            loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
            y = tensor_max(out_data[1])
            metrics = tensor_metrics(data[3].float(),y)
            ft_test.append(metrics[7])
            
            # body type
            loss2 = criterion2(out_data[2].squeeze(),data[4].float())    
            y = tensor_max(out_data[2])
            metrics = boolian_metrics(data[4].float(),y)
            ft_test.append(metrics[3])
            
            # leg
            loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
            y = tensor_max(out_data[3])
            metrics = tensor_metrics(data[5].float(),y)
            ft_test.append(metrics[7])
            
            # foot 
            loss4 = criterion1(out_data[4],data[6].argmax(dim=1))      
            y = tensor_max(out_data[4])
            metrics = tensor_metrics(data[6].float(),y)  
            ft_test.append(metrics[7])
            
            # gender
            loss5 = criterion2(out_data[5].squeeze(),data[7].float())
            y = tensor_max(out_data[5])
            metrics = boolian_metrics(data[7].float(),y)  
            ft_test.append(metrics[3])
            
            # bags
            loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
            y = tensor_max(out_data[6])
            metrics = tensor_metrics(data[8].float(),y)
            ft_test.append(metrics[7])
            
            # body colour
            loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
            y = tensor_max(out_data[7])
            metrics = tensor_metrics(data[9].float(),y)
            ft_test.append(metrics[7])
            
            # leg colour
            loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
            y = tensor_max(out_data[8])
            metrics = tensor_metrics(data[10].float(),y)
            ft_test.append(metrics[7])
            
            # foot colour
            loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
            y = tensor_max(out_data[9])
            metrics = tensor_metrics(data[11].float(),y)      
            ft_test.append(metrics[7])
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
            loss_t.append(loss.item())
    test_loss.append(np.mean(loss_t))
    F1_test.append(np.mean(ft_test))
    print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
    torch.save(test_loss,'/home/hossein/anaconda3/envs/torchreid/my_osnet/testloss_v1_02.pth')
    
    if len(F1_test)>2: 
        if F1_test[-1] > F1_test[-2]:
            print('our net improved')
            torch.save(attr_net , '/home/hossein/anaconda3/envs/torchreid/my_osnet/attrnet_v1_02.pth')
            
                 