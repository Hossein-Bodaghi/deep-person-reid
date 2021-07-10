#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:28:41 2021

@author: hossein

here we can find different types of trainigs 
that are define for person-attribute detection.
this is Hossein Bodaghies thesis
"""
import torch
import numpy as np
from metrics import tensor_metrics,boolian_metrics
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

#%%
'''
*
functions which are needed for training proccess.

tensor_max: takes a matrix and return a matrix with one hot vectors for max argument
lis2tensor: takes a list containing torch tensors and return a torch matrix  
id_onehot: takes id tensors and make them one hoted 
'''
def tensor_max(tensor):

    idx = torch.argmax(tensor, dim=1, keepdim=True)
    y = torch.zeros(tensor.size(),device=device).scatter_(1, idx, 1.)
    return y

def tensor_thresh(tensor, thr=0.5):
    out = (tensor>thr).float()
    return out
        

def list2tensor(list1):
    tensor = torch.zeros((len(list1),list1[0].size()[0]))
    for i in range(len(list1)):
        tensor[i] = list1[i]
    return tensor   

def id_onehot(id_,num_id):
    # one hot id vectors
    id1 = torch.zeros((len(id_),num_id))
    for i in range(len(id1)):
        a = id_[i]
        id1[i,a-1] = 1
    return id1

#%%
def train_collection(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     saving_path,
                     version): 
    # def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
    train_loss = []
    test_loss = []
    F1_train = []
    F1_test = []
    gc.collect()
    torch.cuda.empty_cache()
    attr_net = attr_net.to(device)
    
    for epoch in range(1,num_epoch+1):
        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
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
            y = tensor_thresh(out_data[2], 0.5)
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
            y = tensor_thresh(out_data[5], 0.5)
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
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
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
                y = tensor_thresh(out_data[2], 0.5)
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
                y = tensor_thresh(out_data[5], 0.5)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
                y = tensor_thresh(out_data[7], 0.5)
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
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        
        scheduler.step()
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net , saving_path+'attrnet_'+version+'.pth')
                torch.save(optimizer, saving_path+'optimizer_'+version+'.pth')
                
  
#%%
def train_collection_id(num_epoch,
                     attr_net,
                     train_loader,
                     test_loader,
                     optimizer,
                     scheduler,
                     criterion1,
                     criterion2,
                     saving_path,
                     num_id,
                     device,
                     version,
                     resume=False,
                     loss_train = None,
                     loss_test=None,
                     train_F1=None,
                     test_F1=None,
                     stop_epoch=None): 
    # model_output (tuple): (out_head,out_body,out_body_type,out_leg,out_foot,out_gender,out_bags,out_body_colour,out_leg_colour,out_foot_colour,out_id)
    # loader_outpu (tuple): (img,id,head,body,body_type,leg,foot,gender,bags,body_colour,leg_colour,foot_colour)
    print('this is start')
    if resume:
        train_loss = loss_train
        test_loss = loss_test
        F1_train = train_F1
        F1_test = test_F1
    else:
        train_loss = []
        test_loss = []
        F1_train = []
        F1_test = []
    # gc.collect()
    # torch.cuda.empty_cache()
    # attr_net = attr_net.to(device)
    print('epoches started')
    if resume:
        start_epoch = stop_epoch+1
    else:
        start_epoch = 1    
    for epoch in range(start_epoch,num_epoch+1):
        
        torch.cuda.empty_cache()
        attr_net = attr_net.to(device)        
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        
        for idx, data in enumerate(train_loader):
            
            # forward step
            optimizer.zero_grad()
            out_data = attr_net.forward(data[0].to(device))
            
            for i in range(len(data)):
                data[i] = data[i].to(device)
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
            y = tensor_thresh(out_data[2], 0.5)
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
            y = tensor_thresh(out_data[5], 0.5)
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
            
            # id
            loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
            y = tensor_max(out_data[-1])
            metrics = tensor_metrics(data[1].float(),y)      
            ft_train.append(metrics[7])            
            
            # total loss
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss10
            loss_e.append(loss.item())
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            
            # print log
            if idx % 1 == 0:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} , F1: {:.3f}'.format(
                    epoch, idx , len(train_loader),
                     optimizer.param_groups[0]['lr'],
                      loss.item(),np.mean(ft_train)))
       
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        torch.save(train_loss,saving_path+'trainloss_'+version+'.pth')
        torch.save(F1_train,saving_path+'trainF1_'+version+'.pth')
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                
                # data = data.to(device) 'list' object has no attribute 'to'
                # out_data = attr_net.predict(data[0].to(device))
                out_data = attr_net(data[0].to(device))
                for i in range(len(data)):
                    data[i] = data[i].to(device)
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
                y = tensor_thresh(out_data[2], 0.5)
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
                y = tensor_thresh(out_data[5], 0.5)
                metrics = boolian_metrics(data[7].float(),y)  
                ft_test.append(metrics[3])
                
                # bags
                loss6 = criterion1(out_data[6],data[8].argmax(dim=1))
                y = tensor_max(out_data[6])
                metrics = tensor_metrics(data[8].float(),y)
                ft_test.append(metrics[7])
                
                # body colour
                loss7 = criterion1(out_data[7],data[9].argmax(dim=1))      
                y = tensor_thresh(out_data[7], 0.5)
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

                # id
                loss10 = criterion1(out_data[-1],data[1].argmax(dim=1))
                y = tensor_max(out_data[-1])
                metrics = tensor_metrics(data[1].float(),y)      
                ft_train.append(metrics[7])          
                
                # total loss
                loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
                loss_t.append(loss.item())
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        print('Epoch: {}\ntrain loss: {:.6f}\ntest loss: {:.6f}\n\nF1 train: {:.4f}\nF1 test: {:.4f} '.format(
                    epoch,train_loss[-1],test_loss[-1],F1_train[-1],F1_test[-1]))
        
        torch.save(test_loss,saving_path+'testloss_'+version+'.pth')
        torch.save(F1_test,saving_path+'testF1_'+version+'.pth')
        scheduler.step()
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net, saving_path+'attrnet_'+version+'_epoch'+str(epoch)+'.pth')
                torch.save(optimizer.state_dict(), saving_path+'optimizer_'+version+'_epoch'+str(epoch)+'.pth')


#%%    
def train_attr_id(attr_net,
                  train_loader,
                  test_loader,
                  num_epoch,
                  optimizer,
                  scheduler,
                  criterion1,
                  criterion2,
                  saving_path,
                  num_id,
                  version,
                  id_inc=True):

    # def training(train_loader,test_loader,generator,classifier,num_epoch,optimizer,criterion1,criterion2,scheduler,device):
    train_loss = []
    test_loss = []
    F1_train = []
    F1_test = []  
    Acc_train = []
    Acc_test = []
    
    for epoch in range(1,num_epoch+1):
        
        attr_net.to(device)
        attr_net.train()
        loss_e = []
        loss_t = []
        ft_train = []
        ft_test = []
        acc_train = []
        acc_test = []
        
        for idx, data in enumerate(train_loader):

            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)           
            # forward step
            optimizer.zero_grad()
            out_data = attr_net(data[0])
            #out_data[0] = out_data[0].to(device)
            #out_data[1] = out_data[1].to(device)
            
            # compute losses and evaluation metrics:
                
            # attributes
            loss0 = criterion2(out_data[1],data[2].float())  
            attr_out = torch.round(out_data[1])
            metrics = tensor_metrics(data[2].float(),attr_out)
            ft_train.append(metrics[7])
            
            if id_inc:
                loss1 = criterion1(out_data[0],data[1])  
                y = tensor_max(out_data[0].to('cpu'))
                oh_id = id_onehot(data[1],num_id)
                metrics = tensor_metrics(oh_id.float(),y) 
                acc_train.append(metrics[-2])  
                loss = loss0+loss1
            else:
                loss = loss0
            loss_e.append(loss.item())  
            
            # backward step
            loss.backward()
            
            # optimization step
            optimizer.step()
            scheduler.step()
            # print log
            if id_inc:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \t F1: {:.3f} \t accuracy:{:.3f}'.format(
                    epoch,
                    idx , len(train_loader), 
                    optimizer.param_groups[0]['lr'], 
                    loss_e[-1],
                    np.mean(ft_train),
                    np.mean(acc_train)))
            else:
                print('Train Epoch: {} [{}/{} , lr {}] \t Loss: {:.6f} \t F1: {:.3f}'.format(
                    epoch,
                    idx , len(train_loader), 
                    optimizer.param_groups[0]['lr'], 
                    loss.item(),np.mean(ft_train)))                
        if id_inc:
            Acc_train.append(np.mean(acc_train))
            
        train_loss.append(np.mean(loss_e))
        F1_train.append(np.mean(ft_train))
        
        # evaluation:     
        attr_net.eval()
        with torch.no_grad():
            for idx, data in enumerate(test_loader):

                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                data[2] = data[2].to(device)
                out_data = attr_net(data[0])
                #out_data[0] = out_data[0].to(device)
                #out_data[1] = out_data[1].to(device)
                # compute losses and evaluation metrics:
                    
                # attributes
                loss0 = criterion2(out_data[1],data[2].float())  
                attr_out = torch.round(out_data[1])
                metrics = tensor_metrics(data[2].float(),attr_out)
                ft_test.append(metrics[7])
                
                if id_inc:
                    loss1 = criterion1(out_data[0],data[1])  
                    y = tensor_max(out_data[0])
                    oh_id = id_onehot(data[1],num_id)
                    metrics = tensor_metrics(oh_id.float(),y)
                    acc_test.append(metrics[-2])  
                    loss = loss0+loss1
                else:
                    loss = loss0
                loss_t.append(loss.item())  
        test_loss.append(np.mean(loss_t))
        F1_test.append(np.mean(ft_test))
        if id_inc:
            Acc_test.append(np.mean(acc_test))
            print('Epoch: {} \n train loss: {:.6f} \n test loss: {:.6f} \n \n F1 train: {:.4f} \n F1 test: {:.4f} \n accuracy train: {:.4f} \n accuracy test: {:.4f}'.format(
                        epoch,
                        train_loss[-1],
                        test_loss[-1],
                        F1_train[-1],
                        F1_test[-1],
                        Acc_train[-1],
                        Acc_test[-1]))
        else:
            print('Epoch: {} \n train loss: {:.6f} \n test loss: {:.6f} \n \n F1 train: {:.4f} \n F1 test: {:.4f}'.format(
                        epoch,
                        train_loss[-1],
                        test_loss[-1],
                        F1_train[-1],
                        F1_test[-1]))            
            
        
        torch.save(test_loss,saving_path+saving_path+'testloss_'+version+'.pth')
        torch.save(train_loss,saving_path+saving_path+'trainloss_'+version+'.pth')
        
        torch.save(F1_test,saving_path+'F1_test_'+version+'.pth')
        torch.save(F1_train,saving_path+'F1_train_'+version+'.pth')
        if id_inc:
            torch.save(Acc_test,saving_path+'testacc_'+version+'.pth')
            torch.save(Acc_train,saving_path+'trainacc_'+version+'.pth') 
        if len(F1_test)>2: 
            if F1_test[-1] > F1_test[-2]:
                print('our net improved')
                torch.save(attr_net, saving_path+'attrnet_'+version+'.pth')
                torch.save(optimizer.state_dict(), saving_path+'attrnet_'+version+'.pth')



