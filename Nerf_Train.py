import torch
import torch.nn as nn
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model2 import Voxels,Nerf,FullyFusedNerf
from ml_helpers import training
import rendering
# Load the two images






torch.cuda.empty_cache()    # clear the GPU and Stop all other program running
data_set_path = '/home/eiyike/PhdProjects/NerfPHD/new_code_update1/Dataset'
#data_set_path= '/home/eiyike/PhdProjects/NerfPHD/new_code_update1/ORNLDATASET'




mode = 'train'


target_size = (400,400)
dataset = data_preprocessing(data_set_path,mode,target_size=target_size)


o, d, target_px_values,total_data = dataset.get_rays()

batch_size = 1024


# size_h, size_w = (400,400)
# dataloader= DataLoader(
#     torch.cat(
#         (torch.from_numpy(o).reshape(total_data, size_h, size_w, 3)[:10, :, :, :].reshape(-1, 3).type(torch.float),
#          torch.from_numpy(d).reshape(total_data, size_h, size_w, 3)[:10, :, :, :].reshape(-1, 3).type(torch.float),
#         torch.from_numpy(target_px_values).reshape(total_data, size_h, size_w, 3)[:10,:, :,  :].reshape(-1, 3).type(torch.float)
#          ), dim=1),
#         batch_size=batch_size, shuffle=True)

size_h, size_w = target_size
dataloader_warmup = DataLoader(
    torch.cat(
        (torch.from_numpy(o).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
         torch.from_numpy(d).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
        torch.from_numpy(target_px_values).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3)),
        dim=1),
        batch_size=batch_size, shuffle=True)

dataloader = DataLoader(
    torch.cat(
              (torch.from_numpy(o).reshape(-1, 3).type(torch.float),
               torch.from_numpy(d).reshape(-1, 3).type(torch.float),
               torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)
               ), dim=1),
              batch_size=batch_size, shuffle=True)

device = 'cuda'

tn = 2
tf = 6
nb_epochs = 10
lr = 1e-3
gamma = .5
nb_bins = 100

#model = Nerf(hidden_dim=128).to(device)
model = FullyFusedNerf(hidden_dim=128).to(device)   # For fullyfused network
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, device=device)
plt.plot(training_loss)
plt.show()



# img = rendering.rendering(model, torch.from_numpy(o[0]).type(torch.float32).to(device), torch.from_numpy(d[0]).type(torch.float32).to(device),
#                     tn, tf, nb_bins=100, device=device)
#
# print(img.reshape(400,400,3))
# plt.imshow(img.reshape(400, 400, 3).data.cpu().numpy())