#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:28:19 2021

@author: hossein

* mean average precision function for person re-id by pytorch* 
"""

import torch
from collections import Counter 


def mean_average_precisions(
        pred_ids, true_ids, num_people, threshold=0.5
):
    
    
    average_precisions = [] # each elemnt is for one person
    epsilon = 1e-6
    
    # pred_ids (list): [[image_id, class_pred, prob_score], ...]
    # class_pred (integer): the predicted id
    # prob_score (floot): the id score from softmax or other last layer prediction
    for c in range(num_people):
        detections = []
        ground_truths = []
        
        for detection in pred_ids:
            if detection[1] == c:
                detections.append(detection)
                
        for true_id in true_ids:
            if true_id[1] == c:
                ground_truths.append(true_id)
        
        # amount_ids = {0:3, 1:5}
        # it means first pic has 3 people with same id that is impossible in our case 
        # amount_ids = Counter([gt[0] for gt in ground_truths])
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_ids = len(ground_truths)
        
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                ids for ids in ground_truths if ids[0] == detection[0]
                ]
            num_gts = len(ground_truth_img)
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                pass
            
                
           