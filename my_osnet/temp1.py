# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torchreid
import torchsummary
from pytorch_model_summary import summary

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)


datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='viper',
    targets='viper',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)



print(datamanager.num_train_pids)


from torchreid import models
import torch

# models.show_avai_models()
# model = models.build_model(
#     'osnet_ain_x1_0',
#     751 , loss='softmax',use_gpu=False,
#     pretrained = False
#                         )
model1 = models.osnet_ain.osnet_ain_x1_0(pretrained=False)
model_state_dict = torch.load('/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
model1.state_dict(model_state_dict)
model1.eval()
a = list(model1.children())


# eliminate the module word from begining of each key 
my_dict = dict(model_state_dict)
dict2 = list(my_dict)
for k , v  in my_dict.items():
    b = k.split('.') # split name of each layer by '. 
    b2 = b[1:] # delete the first one 'module'.
    k_new = '.'.join(b2) # join the list strings with a '.' between them.
    print(k_new)
    my_dict[k_new] = my_dict[k]   # change the previous key to new key (without module).
    del my_dict[k]
    print(my_dict[k_new])


    
model_state_dict.items()

model1.load_state_dict(
    model_state_dict
                    )






my_dict = dict(model_state_dict)
my_dict1 = model_state_dict
for k , v in my_dict1.items():
    b = k.split('.')
    b2 = b[1:]
    k_new = '.'.join(b2)
    my_dict1[k] = k_new
    print(my_dict1[k])
# model1 = torch.load('/home/hossein/anaconda3/envs/torchreid/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
#                     , map_location= torch.device("cpu"))
# model1.children()
# import pytorch_model_summary as pms
# pms.summary(model,print_summary=True)


# summary(model)