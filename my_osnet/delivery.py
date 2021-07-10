#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:06:30 2021

@author: hossein
"""
import numpy as np
import torch
import os 


def data_delivery(main_path,
                  path_attr=None,
                  path_start=None,
                  only_id=False,
                  double = True,
                  need_collection=False,
                  need_attr=True):
    '''
    

    Parameters
    ----------
    main_path : TYPE string
        DESCRIPTION. the path of images folder
    path_attr : TYPE numpy array
        DESCRIPTION.
    path_start : TYPE 
        DESCRIPTION.
    double : TYPE true/false
        DESCRIPTION. will double everything and return 
    need_collection : TYPE true/false
        DESCRIPTION. The default is False.
        if it is false returns a tuple containes a list of 
        image_names and their attributes in numpy and a list of ids  
    need_attr : when we want to see the whole attributes as a target vector 
    Returns
    only_id : when you need only id and id_weights. 
    -------
    None.

    '''
    
    if path_attr:
            # loading attributes
        start_point = np.load(path_start)
        attr_vec_np = np.load(path_attr)# loading attributes
    
            # attributes
        attr_vec_np = attr_vec_np.astype(np.int32)
        attr_vec_np = attr_vec_np[:start_point]
        if double:
            attr_vec_np = np.append(attr_vec_np,attr_vec_np,axis=0)
        
        # images names
    
    img_names = os.listdir(main_path)
    img_names.sort()
    if only_id:
        pass
    else:
        img_names = img_names[:start_point]
    img_names = np.array(img_names)
    if double:
        img_names = list(np.append(img_names,img_names,axis=0))
        
        # ids & ids_weights
    id_ = []
    cam_id = []
    for name in img_names:
        b = name.split('_')
        id_.append(int(b[0])-1)
        cam_id.append(int(b[1][1]))
    id_ = torch.from_numpy(np.array(id_))# becuase list doesnt take a list of indexes it should be slice or inegers.
    cam_id = np.array(cam_id)
    # numbers = torch.unique(id_) # return individual numbers in a tensor
    iterations = torch.bincount(id_) # return iterations of each individual number 
    gp = (int(torch.max(iterations))-int(torch.min(iterations)))//5
    min_it = int(torch.min(iterations))
    id_weights = torch.ones(iterations.size())
    
    for j in range(len(id_weights)):
            if min_it<iterations[j]<=min_it+gp:
                id_weights[j] = 5        
            elif min_it+gp<iterations[j]<=min_it+2*gp:
                id_weights[j] = 4
            elif min_it+2*gp<iterations[j]<=min_it+3*gp:
                id_weights[j] = 3
            elif min_it+3*gp<iterations[j]<=min_it+4*gp:
                id_weights[j] = 2
            elif min_it+4*gp<iterations[j]<=int(torch.max(iterations)):
                id_weights[j] = 1 
    
    if only_id:
        return {'img_names':np.array(img_names),'id':id_,'id_weights':id_weights, 'cam_id':cam_id}
        
        # frequency and frequencies weights
    frequencies = np.sum(attr_vec_np,axis=0)
    frequencies = torch.from_numpy(frequencies)
    gp = int(torch.max(frequencies))//5
    freq_weights = torch.zeros(frequencies.size())
    
    for i in range(len(frequencies)):
        if frequencies[i] == 0:
            pass
        elif 0<frequencies[i]<=gp:
            freq_weights[i] = 5        
        elif gp<frequencies[i]<=2*gp:
            freq_weights[i] = 4
        elif 2*gp<frequencies[i]<=3*gp:
            freq_weights[i] = 3
        elif 3*gp<frequencies[i]<=4*gp:
            freq_weights[i] = 2
        elif 4*gp<frequencies[i]<=int(torch.max(frequencies)):
            freq_weights[i] = 1 
            
    if need_attr:
        return {'id':id_,
                'id_weights':id_weights,
                'img_names':np.array(img_names),
                'frequencies':frequencies,
                'freq_weights':freq_weights,
                'attributes':torch.from_numpy(attr_vec_np),
                'cam_id':cam_id} 
    
    if need_collection:
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
        
        for vec in attr_vec_np:
            
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
            
            
        return {'id':id1,
                'id_weights':id_weights,
                'freq_weights':freq_weights,
                'frequencies':frequencies,
                'img_names':np.array(img_names),
                'head':torch.from_numpy(np.array(head)),
                'body':torch.from_numpy(np.array(body)),
                'body_type':torch.tensor(body_type),
                'leg':torch.from_numpy(np.array(leg)),
                'foot':torch.from_numpy(np.array(foot)),
                'gender':torch.tensor(gender),
                'bags':torch.from_numpy(np.array(bags)),
                'body_colour':torch.from_numpy(np.array(body_colour)),
                'leg_colour':torch.from_numpy(np.array(leg_colour)),
                'foot_colour':torch.from_numpy(np.array(foot_colour)),
                'cam_id':cam_id}


        
    
    
