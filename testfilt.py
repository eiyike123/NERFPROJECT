import torch
import torch.nn as nn
import numpy as np

import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model import Voxels,Nerf
from ml_helpers import training
import rendering
import cv2
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import msssim
#%%
torch.cuda.empty_cache()    # clear the GPU and Stop all other program running
data_set_path = '/home/eiyike/PhdProjects/NerfPHD/new_code_update1/Dataset'
#data_set_path= '/home/eiyike/PhdProjects/NerfPHD/new_code_update1/ORNLDATASET'
mode = 'test'
target_size = (400,400)
dataset = data_preprocessing(data_set_path,mode,target_size=target_size)


test_o, test_d, target_px_values,total_data = dataset.get_rays()



device='cuda'
tn=2
tf=6

#%%
model = torch.load('model_nerffully12layer').to(device)
#%%
def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))


@torch.no_grad()
def test(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None):
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)

    image = []
    for o_batch, d_batch in zip(o, d):
        img_batch = rendering.rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
        image.append(img_batch)  # N, 3
    image = torch.cat(image)
    image = image.reshape(H, W, 3).cpu().numpy()
    if target is not None:
        mse=((image-target)**2).mean()
        psnr= mse2psnr(mse)
    if  target is not None:
        return image, mse,psnr
    else:
        return image





    # return image
#%%
# img = test(model, torch.from_numpy(test_o[0]).to(device).float(), torch.from_numpy(test_d[0]).to(device).float(),
#                 tn, tf, nb_bins=100, chunk_size=10)

img,mse,psnr= test(model, torch.from_numpy(test_o[4]).to(device).float(), torch.from_numpy(test_d[4]).to(device).float(),
                tn, tf, nb_bins=100, chunk_size=10,target=target_px_values[4].reshape(400,400,3))






#image generated using nerf
plt.imshow(img)
plt.title("Image Generated by NeRF")
plt.savefig("nerffully3.png")
#plt.show()

print(psnr)
#Original Image from the test data
#real image
plt.imshow(target_px_values[4].reshape(400,400,3))
plt.title("Original Test Image")
#plt.show()
plt.savefig("testimagefully3.png")
#
#difference between the original image and that reconstructed using nerf
plt.imshow(img-target_px_values[4].reshape(400,400,3))
plt.title("Difference between NeRF and Test Image")
#plt.show()
plt.savefig("difference_nerf_test.png")
#
print("The mean Square error is:",mse)
print("The Peak Signal to Noise Ratio(PSNR) is:",psnr)


# #differences and ground truth
# #diff_img=np.abs(img-target_px_values[0].reshape(400,400,3))-
# diff_img=(img-target_px_values[4].reshape(400,400,3))
#
# # # Combine the normalized difference and ground truth images
# # combined_img = 0.8 * diff_img + 0.2 * target_px_values[0].reshape(400,400,3)
# # plt.imshow(combined_img)
# # plt.title("Combine the differences and the ground truth")
# # #plt.show()
# # plt.savefig("combineddiffgt.png")
#
#
#
# original_image= target_px_values[4].reshape(400,400,3)
#
#
# # #ssim
# #
# # ssim= ssim(img,original_image, multichannel=True)
# # print('ssim',ssim)
#
# diff_image = (img- original_image)
#
#
#
# # convert the normalized difference values back to their original pixel values
# t_max = 255
# t_min = 0
# s_max = diff_image.max()
# s_min = diff_image.min()
# diff_image =(diff_image - s_min) / (s_max - s_min) * (t_max - t_min) + t_min # assuming the original image was stored as 8-bit unsigned integers
#
# # create a binary mask of non-zero pixels in the original image
# mask = (original_image > 0).astype(np.uint8)
#
# # apply the mask to the difference image
# masked_difference = diff_image * mask
#
# # plot the masked difference image
# plt.imshow(masked_difference)
# plt.axis('off')
# plt.show()

image1=cv2.imread('nerffully3.png')
image2=cv2.imread('testimagefully3.png')

SSIM=msssim(image1,image2)

print('similarity index:', SSIM)

# similarity index: (0.9481933016112474, 0.9488951652671916)



#NERFFULLY  8 layoers
# The mean Square error is: 0.002368911472530479
# The Peak Signal to Noise Ratio(PSNR) is: 26.254511687875315
# similarity index: (0.974370023187753+0j)

#NERF NERF  8 layers
# The mean Square error is: 0.0022810673722247596
# The Peak Signal to Noise Ratio(PSNR) is: 26.41861887475062
# similarity index: (0.9748056287378575+0j)


#normal ner 256    8 layers

# The mean Square error is: 0.00317976800321634
# The Peak Signal to Noise Ratio(PSNR) is: 24.976045651094125
# similarity index: (0.9694464613840008+0j)



#5layer  NERF
# The mean Square error is: 0.0022159720543275274
# The Peak Signal to Noise Ratio(PSNR) is: 26.544357208063673
# similarity index: (0.9754210472215303+0j)


#3layers   NERF
# The mean Square error is: 0.0027743590567749416
# The Peak Signal to Noise Ratio(PSNR) is: 25.568373333546624
# similarity index: (0.9717726301966257+0j)

#10 layers Nerf



#nerfully 5

# The mean Square error is: 0.002744483134722624
# The Peak Signal to Noise Ratio(PSNR) is: 25.615394336815665
# similarity index: (0.9725760053992859+0j)

#nerffully 3
# The mean Square error is: 0.002744483134722624
# The Peak Signal to Noise Ratio(PSNR) is: 25.615394336815665
# similarity index: (0.9725760053992859+0j)

#Nerffully 10 layers

# The mean Square error is: 0.0020353495676755902
# The Peak Signal to Noise Ratio(PSNR) is: 26.913609907263243
# similarity index: (0.9764334356757552+0j)

#nerffully 12 layers

# The mean Square error is: 0.002102704162609824
# The Peak Signal to Noise Ratio(PSNR) is: 26.772218255466207
# similarity index: (0.9758573063398183+0j)