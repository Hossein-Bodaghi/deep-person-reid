#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:57:38 2021

@author: hossein
"""

from scipy import io
import numpy as np


def PA_100K_loading(path):
    '''
    0)female 1)over60 2)18_60 3)less18 4)front 5)side 6)back 
    7)hat 8)glasses 9)handbag 10)shoulder_bag 11)backpack 
    12)hold_objects_in_fronts 13)short_sleeve 14)long_sleeve 15)upper_stride 
    16)upper_logo 17)upper_plaid 18)upper_splice 19)lower_stripe 20)lower_pattern 
    21)long_coat 22)trousers 23)shorts 24)skirt&dress 25)boots
    
    attributes_names: a list 26
    train_images_names: a list 80000 (names of training set images)
    test_images_names: a list 10000 (names of testing set images)
    val_images_names: a list 10000 (names of validation set images)
    
    train_label: numpy array (80000,26)
    test_label: numpy array (10000,26)
    val_label: numpy array (10000,26)
    '''
    mat = io.loadmat(path)
    
    attributes = mat['attributes']
    attributes_names = []
    for key in attributes:
        if str(np.ndarray.tolist(key))[-18] == "'":
            attributes_names.append(str(np.ndarray.tolist(key))[9:-18])
        else:
            attributes_names.append(str(np.ndarray.tolist(key))[9:-17])
    
    test_images_name = mat['test_images_name']
    test_images_names = []
    for key in test_images_name:
        if str(np.ndarray.tolist(key))[-18] == "'":
            test_images_names.append(str(np.ndarray.tolist(key))[9:-18])
        else:
            test_images_names.append(str(np.ndarray.tolist(key))[9:-17])
                
    train_images_name = mat['train_images_name']
    train_images_names = []
    for key in train_images_name:
        if str(np.ndarray.tolist(key))[-18] == "'":
            train_images_names.append(str(np.ndarray.tolist(key))[9:-18])
        else:
            train_images_names.append(str(np.ndarray.tolist(key))[9:-17])
    
    val_images_name = mat['val_images_name']
    val_images_names = []
    for key in val_images_name:
        if str(np.ndarray.tolist(key))[-18] == "'":
            val_images_names.append(str(np.ndarray.tolist(key))[9:-18])
        else:
            val_images_names.append(str(np.ndarray.tolist(key))[9:-17])
            
    train_label = mat['train_label']        
    test_label = mat['test_label']        
    val_label = mat['val_label']
    
    return {'attributes_names':attributes_names, 'train_images_names':train_images_names,
            'test_images_names':test_images_names, 'val_images_names':val_images_names,
            'train_label':train_label, 'test_label':test_label, 'val_label':val_label}

path = '/home/hossein/deep-person-reid/datasets/PA-100K/annotation/annotation.mat'
attr = PA_100K_loading(path)