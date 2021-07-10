#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:23:30 2021

@author: hossein

this is first version of training on market 1501 attribute dettection

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

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
test_loader = DataLoader(test_data,batch_size=10,shuffle=True)

#%%
'''
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
loss0,loss1,loss2,loss3,loss4 = [],[],[],[],[]
loss5,loss6,loss7,loss8,loss9 = [],[],[],[],[]

for epoch in range(1,num_epoch+1):
    
    attr_net.train()
    loss_e = []
    
    # categories losses
    loss0_e,loss1_e,loss2_e,loss3_e,loss4_e = [],[],[],[],[]
    loss5_e,loss6_e,loss7_e,loss8_e,loss9_e = [],[],[],[],[]
    
    # categories precisions
    pt0,pt1,pt2,pt3,pt4 = [],[],[],[],[]
    pt5,pt6,pt7,pt8,pt9 = [],[],[],[],[]

    # tags precisions
    p0,p1,p3,p4 = [],[],[],[],[]
    p5,p6,p7,p8,p9 = [],[],[],[],[]
    
    # categories recalls
    rt0,rt1,rt2,rt3,rt4 = [],[],[],[],[]
    rt5,rt6,rt7,rt8,rt9 = [],[],[],[],[]
    
    # tags recalls
    r0,r1,r3,r4 = [],[],[],[],[]
    r5,r6,r7,r8,r9 = [],[],[],[],[]
    
    # categories accuracies
    at0,at1,at2,at3,at4 = [],[],[],[],[]
    at5,at6,at7,at8,at9 = [],[],[],[],[]
    
    # tags accuracies
    a0,a1,a3,a4 = [],[],[],[],[]
    a5,a6,a7,a8,a9 = [],[],[],[],[]
    
    # categories F1
    ft0,ft1,ft2,ft3,ft4 = [],[],[],[],[]
    ft5,ft6,ft7,ft8,ft9 = [],[],[],[],[]
    
    # tags F1
    f0,f1,f3,f4 = [],[],[],[],[]
    f5,f6,f7,f8,f9 = [],[],[],[],[]
    
    for idx, data in enumerate(train_loader):
        
        # forward step
        optimizer.zero_grad()
        out_data = attr_net(data[0])
        
        # compute losses and evaluation metrics:
            
        # head 
        loss0 = criterion1(out_data[0],data[2].argmax(dim=1))
        loss0_e.append(loss0.item())
        
        y = tensor_max(out_data[0])
        metrics0 = tensor_metrics(data[2].float(),y)
        p0.append(metrics0[0])
        r0.append(metrics0[1])
        a0.append(metrics0[2])
        f0.append(metrics0[3])
        pt0.append(metrics0[4])
        rt0.append(metrics0[5])
        at0.append(metrics0[6])
        ft0.append(metrics0[7])
        
        
        # body
        loss1 = criterion1(out_data[1],data[3].argmax(dim=1))
        loss1_e.append(loss1.item())
        
        y = tensor_max(out_data[1])
        metrics1 = tensor_metrics(data[3].float(),y)
        p1.append(metrics1[0])
        r1.append(metrics1[1])
        a1.append(metrics1[2])
        f1.append(metrics1[3])
        pt1.append(metrics1[4])
        rt1.append(metrics1[5])
        at1.append(metrics1[6])
        ft1.append(metrics1[7])
        
        # body type
        loss2 = criterion2(out_data[2].squeeze(),data[4].float())
        loss2_e.append(loss2.item())
        
        y = tensor_max(out_data[2])
        metrics2 = boolian_metrics(data[4].float(),y)
        pt2.append(metrics2[0])
        rt2.append(metrics2[1])
        at2.append(metrics2[2])
        ft2.append(metrics2[3])
        
        # leg
        loss3 = criterion1(out_data[3],data[5].argmax(dim=1))
        loss3_e.append(loss3.item())
        
        y = tensor_max(out_data[3])
        metrics3 = tensor_metrics(data[5].float(),y)
        p3.append(metrics3[0])
        r3.append(metrics3[1])
        a3.append(metrics3[2])
        f3.append(metrics3[3])
        pt3.append(metrics3[4])
        rt3.append(metrics3[5])
        at3.append(metrics3[6])
        ft3.append(metrics3[7])
        
        # foot 
        loss4 = criterion1(out_data[4],data[6].argmax(dim=1))
        loss4_e.append(loss4.item())
        
        y = tensor_max(out_data[4])
        metrics4 = tensor_metrics(data[6].float(),y)  
        p4.append(metrics4[0])
        r4.append(metrics4[1])
        a4.append(metrics4[2])
        f4.append(metrics4[3])
        pt4.append(metrics4[4])
        rt4.append(metrics4[5])
        at4.append(metrics4[6])
        ft4.append(metrics4[7])
        
        # gender
        loss5 = criterion2(out_data[5].squeeze(),data[7].float())
        loss5_e.append(loss5.item())
        
        y = tensor_max(out_data[5])
        metrics5 = boolian_metrics(data[7].float(),y)  
        pt5.append(metrics5[0])
        rt5.append(metrics5[1])
        at5.append(metrics5[2])
        ft5.append(metrics5[3])
        
        # bags
        loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
        loss6_e.append(loss6.item())
        
        y = tensor_max(out_data[6])
        metrics6 = tensor_metrics(data[8].float(),y)
        p6.append(metrics6[0])
        r6.append(metrics6[1])
        a6.append(metrics6[2])
        f6.append(metrics6[3])
        pt6.append(metrics6[4])
        rt6.append(metrics6[5])
        at6.append(metrics6[6])
        ft6.append(metrics6[7])
        
        # body colour
        loss7 = criterion1(out_data[7],data[9].argmax(dim=1))
        loss7_e.append(loss7.item())
        
        y = tensor_max(out_data[7])
        metrics7 = tensor_metrics(data[9].float(),y)
        p7.append(metrics7[0])
        r7.append(metrics7[1])
        a7.append(metrics7[2])
        f7.append(metrics7[3])
        pt7.append(metrics7[4])
        rt7.append(metrics7[5])
        at7.append(metrics7[6])
        ft7.append(metrics7[7])
        
        # leg colour
        loss8 = criterion1(out_data[8],data[10].argmax(dim=1))
        loss8_e.append(loss8.item())
        
        y = tensor_max(out_data[8])
        metrics8 = tensor_metrics(data[10].float(),y)
        p8.append(metrics8[0])
        r8.append(metrics8[1])
        a8.append(metrics8[2])
        f8.append(metrics8[3])
        pt8.append(metrics8[4])
        rt8.append(metrics8[5])
        at8.append(metrics8[6])
        ft8.append(metrics8[7])
        
        # foot colour
        loss9 = criterion1(out_data[9],data[11].argmax(dim=1))
        loss9_e.append(loss9.item())
        
        y = tensor_max(out_data[9])
        metrics9 = tensor_metrics(data[11].float(),y)      
        p9.append(metrics9[0])
        r9.append(metrics9[1])
        a9.append(metrics9[2])
        f9.append(metrics9[3])
        pt9.append(metrics9[4])
        rt9.append(metrics9[5])
        at9.append(metrics9[6])
        ft9.append(metrics9[7])
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
            print('Train Epoch: {} [{}/{} , idx {}, lr {}] \t Loss: {:.6f}'.format(
                epoch, idx * len(data[8]), len(train_loader), idx, optimizer.param_groups[0]['lr'], loss.item()))
   
    train_loss.append(np.mean(loss_e))
    
    # h    leg_metrics['collection']['loss'].append(np.mean(loss3_e))
    leg_metrics['collection']['precision'].append(np.mean(pt3))
    leg_metrics['collection']['recall'].append(np.mean(rt3))
    leg_metrics['collection']['accuracy'].append(np.mean(at3))
    leg_metrics['collection']['F1'].append(np.mean(ft3))
    p3 = list2tensor(p3)
    leg_metrics['tags']['precision'].append(p3.mean(dim=0))
    r3 = list2tensor(r3)
    leg_metrics['tags']['recall'].append(r3.mean(dim=0))
    a3 = list2tensor(a3)
    leg_metrics['tags']['accuracy'].append(a3.mean(dim=0))
    f3 = list2tensor(f3)
    leg_metrics['tags']['precision'].append(f3.mean(dim=0))
    head_metrics['collection']['loss'].append(np.mean(loss0_e))
    head_metrics['collection']['precision'].append(np.mean(pt0))
    head_metrics['collection']['recall'].append(np.mean(rt0))
    head_metrics['collection']['accuracy'].append(np.mean(at0))
    head_metrics['collection']['F1'].append(np.mean(ft0))
    p0 = list2tensor(p0)
    head_metrics['tags']['precision'].append(p0.mean(dim=0))
    r0 = list2tensor(r0)
    head_metrics['tags']['recall'].append(r0.mean(dim=0))
    a0 = list2tensor(a0)
    head_metrics['tags']['accuracy'].append(a0.mean(dim=0))
    f0 = list2tensor(f0)
    head_metrics['tags']['precision'].append(f0.mean(dim=0))
    
    # body 
    body_metrics['collection']['loss'].append(np.mean(loss1_e))
    body_metrics['collection']['precision'].append(np.mean(pt1))
    body_metrics['collection']['recall'].append(np.mean(rt1))
    body_metrics['collection']['accuracy'].append(np.mean(at1))
    body_metrics['collection']['F1'].append(np.mean(ft1))
    p1 = list2tensor(p1)
    body_metrics['tags']['precision'].append(p1.mean(dim=0))
    r1 = list2tensor(r1)
    body_metrics['tags']['recall'].append(r1.mean(dim=0))
    a1 = list2tensor(a1)
    body_metrics['tags']['accuracy'].append(a1.mean(dim=0))
    f1 = list2tensor(f1)
    body_metrics['tags']['precision'].append(f1.mean(dim=0))    
    
    # body_type
    body_type_metrics['collection']['loss'].append(np.mean(loss2_e))
    body_type_metrics['collection']['precision'].append(np.mean(pt2))
    body_type_metrics['collection']['recall'].append(np.mean(rt2))
    body_type_metrics['collection']['accuracy'].append(np.mean(at2))
    body_type_metrics['collection']['F1'].append(np.mean(ft2))   
    
    # leg
    leg_metrics['collection']['loss'].append(np.mean(loss3_e))
    leg_metrics['collection']['precision'].append(np.mean(pt3))
    leg_metrics['collection']['recall'].append(np.mean(rt3))
    leg_metrics['collection']['accuracy'].append(np.mean(at3))
    leg_metrics['collection']['F1'].append(np.mean(ft3))
    p3 = list2tensor(p3)
    leg_metrics['tags']['precision'].append(p3.mean(dim=0))
    r3 = list2tensor(r3)
    leg_metrics['tags']['recall'].append(r3.mean(dim=0))
    a3 = list2tensor(a3)
    leg_metrics['tags']['accuracy'].append(a3.mean(dim=0))
    f3 = list2tensor(f3)
    leg_metrics['tags']['precision'].append(f3.mean(dim=0))

    # foot
    foot_metrics['collection']['loss'].append(np.mean(loss4_e))
    foot_metrics['collection']['precision'].append(np.mean(pt4))
    foot_metrics['collection']['recall'].append(np.mean(rt4))
    foot_metrics['collection']['accuracy'].append(np.mean(at4))
    foot_metrics['collection']['F1'].append(np.mean(ft4))
    p4 = list2tensor(p4)
    foot_metrics['tags']['precision'].append(p4.mean(dim=0))
    r4 = list2tensor(r4)
    foot_metrics['tags']['recall'].append(r4.mean(dim=0))
    a4 = list2tensor(a4)
    foot_metrics['tags']['accuracy'].append(a4.mean(dim=0))
    f4 = list2tensor(f4)
    foot_metrics['tags']['precision'].append(f4.mean(dim=0))      

    # gender
    gender_metrics['collection']['loss'].append(np.mean(loss5_e))
    gender_metrics['collection']['precision'].append(np.mean(pt5))
    gender_metrics['collection']['recall'].append(np.mean(rt5))
    gender_metrics['collection']['accuracy'].append(np.mean(at5))
    gender_metrics['collection']['F1'].append(np.mean(ft5))

    # bags
    bags_metrics['collection']['loss'].append(np.mean(loss6_e))
    bags_metrics['collection']['precision'].append(np.mean(pt6))
    bags_metrics['collection']['recall'].append(np.mean(rt6))
    bags_metrics['collection']['accuracy'].append(np.mean(at6))
    bags_metrics['collection']['F1'].append(np.mean(ft6))
    p6 = list2tensor(p6)
    bags_metrics['tags']['precision'].append(p6.mean(dim=0))
    r6 = list2tensor(r6)
    bags_metrics['tags']['recall'].append(r6.mean(dim=0))
    a6 = list2tensor(a6)
    bags_metrics['tags']['accuracy'].append(a6.mean(dim=0))
    f6 = list2tensor(f6)
    bags_metrics['tags']['precision'].append(f6.mean(dim=0))    

    # body colour
    body_colour_metrics['collection']['loss'].append(np.mean(loss7_e))
    body_colour_metrics['collection']['precision'].append(np.mean(pt7))
    body_colour_metrics['collection']['recall'].append(np.mean(rt7))
    body_colour_metrics['collection']['accuracy'].append(np.mean(at7))
    body_colour_metrics['collection']['F1'].append(np.mean(ft7))
    p7 = list2tensor(p7)
    body_colour_metrics['tags']['precision'].append(p7.mean(dim=0))
    r7 = list2tensor(r7)
    body_colour_metrics['tags']['recall'].append(r7.mean(dim=0))
    a7 = list2tensor(a7)
    body_colour_metrics['tags']['accuracy'].append(a7.mean(dim=0))
    f7 = list2tensor(f7)
    body_colour_metrics['tags']['precision'].append(f7.mean(dim=0))  

    # leg colour
    leg_colour_metrics['collection']['loss'].append(np.mean(loss8_e))
    leg_colour_metrics['collection']['precision'].append(np.mean(pt8))
    leg_colour_metrics['collection']['recall'].append(np.mean(rt8))
    leg_colour_metrics['collection']['accuracy'].append(np.mean(at8))
    leg_colour_metrics['collection']['F1'].append(np.mean(ft8))
    p8 = list2tensor(p8)
    leg_colour_metrics['tags']['precision'].append(p8.mean(dim=0))
    r8 = list2tensor(r8)
    leg_colour_metrics['tags']['recall'].append(r8.mean(dim=0))
    a8 = list2tensor(a8)
    leg_colour_metrics['tags']['accuracy'].append(a8.mean(dim=0))
    f8 = list2tensor(f8)
    leg_colour_metrics['tags']['precision'].append(f8.mean(dim=0))    

    # foot colour
    foot_colour_metrics['collection']['loss'].append(np.mean(loss9_e))
    foot_colour_metrics['collection']['precision'].append(np.mean(pt9))
    foot_colour_metrics['collection']['recall'].append(np.mean(rt9))
    foot_colour_metrics['collection']['accuracy'].append(np.mean(at9))
    foot_colour_metrics['collection']['F1'].append(np.mean(ft9))
    p9 = list2tensor(p9)
    foot_colour_metrics['tags']['precision'].append(p9.mean(dim=0))
    r9 = list2tensor(r9)
    foot_colour_metrics['tags']['recall'].append(r9.mean(dim=0))
    a9 = list2tensor(a9)
    foot_colour_metrics['tags']['accuracy'].append(a9.mean(dim=0))
    f9 = list2tensor(f9)
    foot_colour_metrics['tags']['precision'].append(f9.mean(dim=0))  

    # evaluation:     
    attr_net.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
        
                 
#%%
# b = nn.functional.one_hot(out_data[0],num_classes=5)       
# idx = torch.argmax(out_data[0], dim=1, keepdim=True)
# y = torch.zeros(out_data[0].size()).scatter_(1, idx, 1.)
    #     # print log
    #     if idx % 1 == 0:
    #         print('Train Epoch: {} [{}/{} , idx {}, lr {}] \t Loss: {:.6f}'.format(
    #             epoch, idx * len(target_image), len(train_loader), idx, optimizer.param_groups[0]['lr'], loss.item()))
            
    #         loss2_e.append(loss.item())
    # train_loss.append(np.mean(loss2_e))
    
    
    # print(train_loss[-1])
    # torch.save(train_loss , '/home/usrx/pretrained/trainloss.pth')
    # # if epoch%10 == 0:
    # #     torch.save(train_loss , '/home/usrx/pretrained/atrainloss_epoch2{}.pth'.format(epoch))
    # #     torch.save(generator , '/home/usrx/pretrained/agenerator_epoch2{}.pth'.format(epoch))
    # #     torch.save(classifier , '/home/usrx/pretrained/alassifier_epoch2{}.pth'.format(epoch))
    # loss2_t = []
    # generator.eval()
    # classifier.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    # with torch.no_grad():
    #     for idx, data in enumerate(test_loader):
    #         target_image, input_person, input_emotion, input_transform = data
    #         target_image = target_image.to(device)
    #         input_person = input_person.to(device)
    #         input_emotion = input_emotion.to(device)
    #         input_transform = input_transform.to(device)
    #         # generator
    #         out_image = generator(input_person, input_emotion, input_transform)
    #         # compute the generator loss 
    #         loss1 = criterion1(out_image, torch.reshape(target_image,(out_image.size()[0],3,158,158)))
    #         # classifier
    #         out_emotion, out_person, out_transform =classifier(out_image)
    #         # compute the classifier loss 
    #         # loss_e = criterion2(out_emotion, input_emotion)
    #         # loss_p = criterion2(out_person, input_person)
    #         # loss_t = criterion2(out_transform, input_transform)
    #         # total loss
    #         loss = loss1
    #         if idx % 1 == 0:
    #             print('Validate: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 idx * len(target_image), len(test_loader),
    #                 100. * idx / len(test_loader), loss.item()))
    #             loss2_t.append(loss.item())
    #     test_loss.append(np.mean(loss2_t))
    #     print(test_loss[-1])   
    #     torch.save(test_loss , '/home/usrx/pretrained/testloss.pth')
    #     loss_min = 1
    #     if test_loss[-1] <= loss_min:
    #         loss_min = test_loss[-1]
    #         torch.save(generator , '/home/usrx/pretrained/generator.pth')
    #         torch.save(classifier , '/home/usrx/pretrained/classifier.pth')



# #%%
# import torch.nn as nn

# class TestNet(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.codef list2tensor(list1):
    # tensor = torch.zeros((len(list1),list1[0].size()[0]))
    # for i in range(len(list1)):
    #     tensor[i] = list1[i]
    # return tensornv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=(3,3))
#         self.pool1 = nn.MaxPool2d((3,3),(2,2))
#         self.LRN = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)  # section 3.3
#     def forward(self,x):
#         out1 = self.conv1(x)
#         out2 = self.pool1(out1)
#         out3 = self.LRN(out2)
#         return (out1,out2,out3)

# transform = transforms.Compose([transforms.RandomRotation(degrees=15),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.RandomVerticalFlip(),
#                                 transforms.RandomPerspective(),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# testmodel = TestNet()
# data_test = MarketLoader(img_path=main_path,attr=attr,resolution=(256,128),transform=transform,indexes=train_idx)  
# a =  data_test[100]
# b = testmodel(a)
# #%%
# import torch
# import torch.nn as nn


# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# print('input is:',input,'\n','target is :',target)

# sigmoid1 = nn.Sigmoid()

# loss1 = nn.BCELoss()

# out1 = sigmoid1(input)
# output_sigmoid1 = loss1(out1, target)
# output_sigmoid1.backward()    
# print('output of sigmoid is:',out1,'\n','output of BCEloss is :',output_sigmoid1)    

# #%%
# '''
# loading label & picture tensors 
# start_point: the last data we labeled
# attr_vec_torch: our attributes 

# '''

# main_path = '/home/hossein/reid-data/market1501/Market-1501-v15.09.15/gt_bbox/'
# path_attr = '/home/hossein/market1501_label/final_attr.npy'
# path_start = '/home/hossein/market1501_label/final_stop.npy'

# # loading attributes
# start_point = np.load(path_start)
# attr_vec_np = np.load(path_attr)
# attr_vec_np = attr_vec_np.astype(np.float32)
# attr_vec_torch = torch.from_numpy(attr_vec_np)
# attr_vec_torch = attr_vec_torch[:start_point]

# # loading 
# transform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                 transforms.RandomVerticalFlip(),
#                                 transforms.RandomPerspective(),
#                                 transforms.ColorJitter(),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# train_data = MarketLoader(main_path,attr_vec_torch,(256,128),transform=transform)


# #%%

# model = models.build_model(
#     name='osnet_ain_x1_0',
#     num_classes=55,
#     loss='softmax',
#     pretrained=True
# )

# new_state_dict = model.state_dict()
# keys = []
# for name in new_state_dict:
#     keys.append(name)
# for idx, m in enumerate(model.children()):
#     print(idx, '->', m)    
# #%%

# class MyOsNet(nn.Module):
#     def __init__(self,model):
#         super().__init__()
#         self.model = model
        
#     def forward(self, x):
#         out = self.model(x)
#         return out
# #%%

