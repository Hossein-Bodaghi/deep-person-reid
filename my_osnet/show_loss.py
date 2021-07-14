import torch
#import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def show_loss(name,train_path,test_path,device):

    im = torch.load(test_path,map_location= torch.device(device))
    im2 = torch.load(train_path,map_location= torch.device(device))
    plt.figure('train')
    plt.legend([im2,im],['train_'+name , 'val_'+name])
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.plot(im2)
    plt.plot(im)

    plt.show()

#%%

train_loss_path = '/home/hossein/deep-person-reid/dr_tale/result/V1_4/trainloss_V1_4.pth'
test_loss_path = '/home/hossein/deep-person-reid/dr_tale/result/V1_4/testloss_V1_4.pth'
train_F1_path = '/home/hossein/deep-person-reid/dr_tale/result/V1_4/trainF1_V1_4.pth'
test_F1_path = '/home/hossein/deep-person-reid/dr_tale/result/V1_4/testF1_V1_4.pth'
show_loss(name='loss', train_path=train_loss_path, test_path=test_loss_path, device=device)
show_loss(name='F1', train_path=train_F1_path, test_path=test_F1_path, device=device)

