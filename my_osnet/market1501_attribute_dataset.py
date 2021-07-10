#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:56:26 2021

@author: hossein
"""


from scipy import io
import numpy as np
 
    
def market_duke_attr(path, key = 'market_attribute'):
    '''
    key = 'gallery' or 'market_attribute' or 'duke_attribute' 
    
    for market1501:
        0) gender 1)hair_length 2)sleeve length 3)lower-body_length 
        4)lower-body_type 5)hat 6)backpack 7)bag 8)hand_bag 9)age[1:4] 
        10)upblack 11)upwhite 12)upred 13)uppurple 14)upyellow 
        15)upgray 16)upblue 17)upgreen 18)downblack 19)downwhite 
        20)downpink 21)downpurple 22)downyellow 23)downgray 24)downblue 
        25)downgreen 26)downbrown 
    
    for DukeMTMC:
        0) gender 1)upper-body_length 2)boots 3)hat 4)backpack 5)bag 
        6)hand_bag 7)age 8)upblack 9)upwhite 10)upred 11)uppurple 12)upgray 
        13)upblue 14)upgreen 15)upbrown 16)downblack 17)downwhite 18)downred 
        19)downgray 20)downblue 21)downgreen 22)downbrown
    '''
    
    mat = io.loadmat(path)
    attr1 = mat[key]
    a = np.ndarray.tolist(attr1)[0][0]
    
    b = np.ndarray.tolist(a[0])[0][0] # a list contains 28 numpy array with the size of 750 (train) for market 
    c = np.ndarray.tolist(a[1])[0][0] # a list contains 28 numpy array with the size of 751 (test) for market
    
    tr_attr = np.squeeze(np.array(b))
    te_attr = np.squeeze(np.array(c))
    
    for i in range(np.shape(tr_attr)[1]):
        tr_attr[-1,i] = int(tr_attr[-1,i])    

    for i in range(np.shape(te_attr)[1]):
        te_attr[-1,i] = int(te_attr[-1,i])
        
    return (tr_attr, te_attr)

# train_path = '/home/hossein/deep-person-reid/market1501_label/Market-1501_Attribute-master/market_attribute.mat'        
# a = market_duke_attr(train_path)    


# attr_path = '/home/hossein/deep-person-reid/datasets/dukemtmc/DukeMTMC-attribute-master/duke_attribute.mat'
# a = market_duke_attr(attr_path , key = 'duke_attribute')

