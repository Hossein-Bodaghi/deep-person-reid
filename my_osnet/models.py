#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:03:10 2021

@author: hossein

here we can find different types of models 
that are define for person-attribute detection. 
this is Hossein Bodaghies thesis 
"""

'''
*

when load a pretrained model from torchreid it just brings imagenet trained models
so if we want to bring pretrained on other datasets we should use this function

'''

from collections import OrderedDict
import torch.nn as nn
from torch import load
import torch

def feature_model(model):
    new_model1 = nn.Sequential(*list(model.children())[:-2])
    return new_model1

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

class MyOsNet(nn.Module):
    
    '''
    this is our network in this version it just take output from features of
    original omni-scale network.
    
    if attr_inc=True then for each attribute has a seperate linear 
    layer for classification
    
    if id_inc=True the output of attribute detection and models features will be concatanated
    and then a clasiification will predict the id of input picture
    '''
    
    def __init__(self,
                 model,
                 num_id,
                 feature_dim=512,
                 attr_dim=55,
                 id_inc=True,
                 attr_inc=True):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batchnormalization = nn.BatchNorm1d(num_features=feature_dim)
        self.dropout = nn.Dropout(0.3)


        self.model = model
        self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim)
        self.id_lin = nn.Linear(in_features=feature_dim+attr_dim , out_features=num_id)
        self.head_lin = nn.Linear(in_features=feature_dim , out_features=5)
        self.body_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_type_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.leg_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.foot_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.gender_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.bags_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.leg_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.foot_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)       
        self.attr_lin = nn.Linear(in_features=feature_dim , out_features=attr_dim)  
        
        self.id_inc = id_inc
        self.attr_inc = attr_inc
        
    def forward(self, x):
        
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.softmax(self.attr_lin(features))
            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
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
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
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
        
    def predict(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.softmax(self.attr_lin(features))
#            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
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
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
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

class MyOsNet2(nn.Module):
    
    '''
    this is our network in this version it just take output from features of
    original omni-scale network.
    
    if attr_inc=True then for each attribute has a seperate linear 
    layer for classification
    
    if id_inc=True the output of attribute detection and models features will be concatanated
    and then a clasiification will predict the id of input picture
    
    in this version forward function and predict function defined seperatetly 
    in forward we dont have 
    '''
    
    def __init__(self,
                 model,
                 num_id,
                 feature_dim=512,
                 attr_dim=46,
                 id_inc=True,
                 attr_inc=True):
        
        super().__init__()
        self.feature_dim = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.batchnormalization = nn.BatchNorm1d(num_features=feature_dim)
        self.dropout = nn.Dropout(0.3)


        self.model = model
        self.linear = nn.Linear(in_features=feature_dim , out_features=feature_dim,)
        self.id_lin = nn.Linear(in_features=feature_dim+attr_dim , out_features=num_id)
        self.head_lin = nn.Linear(in_features=feature_dim , out_features=5)
        self.body_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_type_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.leg_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.foot_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.gender_lin = nn.Linear(in_features=feature_dim , out_features=1)
        self.bags_lin = nn.Linear(in_features=feature_dim , out_features=3)
        self.body_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.leg_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)
        self.foot_colour_lin = nn.Linear(in_features=feature_dim , out_features=9)       
        self.attr_lin = nn.Linear(in_features=feature_dim , out_features=attr_dim)  
        
        self.id_inc = id_inc
        self.attr_inc = attr_inc
        
    def get_feature(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.linear(features)
        return features
        
        
    def forward(self, x):
        
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            out_attr = self.attr_lin(features)

            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.id_lin(concated)
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
            out_head = self.head_lin(features)
            out_body = self.body_lin(features)
            out_body_type = self.body_type_lin(features)
            out_leg = self.leg_lin(features)
            out_foot = self.foot_lin(features)
            out_gender = self.body_type_lin(features)
            out_bags = self.bags_lin(features)
            out_body_colour = self.body_colour_lin(features)
            out_leg_colour = self.leg_colour_lin(features)
            out_foot_colour = self.foot_colour_lin(features)
            
            if self.id_inc:
                out_head1 = self.softmax(out_head)
                out_body1 = self.softmax(out_body)
                out_body_type1 = self.sigmoid(out_body_type)
                out_leg1 = self.softmax(out_leg)
                out_foot1 = self.softmax(out_foot)
                out_gender1 = self.sigmoid(out_gender)
                out_bags1 = self.softmax(out_bags)
                out_body_colour1 = self.softmax(out_body_colour)
                out_leg_colour1 = self.softmax(out_leg_colour)
                out_foot_colour1 = self.softmax(out_foot_colour)  
                
                concated = torch.cat((features,
                                      out_head1,
                                      out_body1,
                                      out_body_type1,
                                      out_leg1,
                                      out_foot1,
                                      out_gender1,
                                      out_bags1,
                                      out_bags1,
                                      out_body_colour1,
                                      out_leg_colour1,
                                      out_foot_colour1),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                # print('the size of out_body_colour layer',out_body_colour.size())
                # print('the size of features layer',features.size())
                # print('the size of concatanated layer',concated.size())
                out_id = self.id_lin(concated)
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:

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
        
    def predict(self, x):
        features = self.model(x)
        features = features.view(-1,self.feature_dim)
        features = self.dropout(features)
        features = self.linear(features)
        features = self.batchnormalization(features)
        features = self.leakyrelu(features)
        features = self.dropout(features)
        
        if self.attr_inc:
            
            # we didnt put any activation becuase regression doesnt need any activation (mse loss can be ok)
            out_attr = self.attr_lin(features)
#            out_attr = self.dropout(out_attr)
            if self.id_inc:
                concated = torch.cat((features,out_attr),dim=1) # dim 1 because all the tensors dimentions start with (batch,..)            
                out_id = self.softmax(self.id_lin(concated))
                return (out_id,out_attr)
            else: 
                return out_attr
        
        else:
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
            
            if self.id_inc:
                concated = torch.cat((features,
                                      out_head,
                                      out_body,
                                      out_body_type,
                                      out_leg,
                                      out_foot,
                                      out_gender,
                                      out_bags,
                                      out_bags,
                                      out_body_colour,
                                      out_leg_colour,
                                      out_foot_colour),dim=1) # the first parameter of torch.cat() should be checked that takes list or tuple or what
                
                concated = self.dropout(concated)
                out_id = self.softmax(self.id_lin(concated))
                return (out_head,
                         out_body,
                         out_body_type,
                         out_leg,
                         out_foot,
                         out_gender,
                         out_bags,
                         out_body_colour,
                         out_leg_colour,
                         out_foot_colour,
                         out_id)
            else:
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