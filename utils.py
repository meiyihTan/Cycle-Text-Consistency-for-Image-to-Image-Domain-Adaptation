import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from PIL import Image
import torchvision
# from torch.autograd import Variable
from CRAFTpytorch import imgproc

def MAELoss(out_image, gt_image):
    return torch.mean(torch.abs(out_image - gt_image))

def SSIMLoss(out_image, gt_image):
    return 1 - ssim(out_image, gt_image, data_range=1, size_average=True)

def MS_SSIMLoss(out_image, gt_image):
    return 1 - ms_ssim(out_image, gt_image, data_range=1, size_average=True)

def PerceptualLoss(out_image_features, gt_image_features):
    pl1 = F.mse_loss(out_image_features.relu1_2, gt_image_features.relu1_2)
    pl2 = F.mse_loss(out_image_features.relu2_2, gt_image_features.relu2_2)
    pl3 = F.mse_loss(out_image_features.relu3_4, gt_image_features.relu3_4)
    pl4 = F.mse_loss(out_image_features.relu4_4, gt_image_features.relu4_4)
    pl5 = F.mse_loss(out_image_features.relu5_4, gt_image_features.relu5_4)
    return (pl1 + pl2 + pl3 + pl4 + pl5) / 5

def TextDetectionLoss(out_image, gt_image, net):
    out_image = torch.clamp(out_image, min=0, max=1)
    out_image = out_image * 255
    out_image = imgproc.normalizeMeanVarianceTensor(out_image)
    
    gt_image = torch.clamp(gt_image, min=0, max=1)
    gt_image = gt_image * 255
    gt_image = imgproc.normalizeMeanVarianceTensor(gt_image)

    out_pred, out_feature = net(out_image)
    gt_pred, gt_feature = net(gt_image)

    out_text = out_pred[:,:,:,0]
    out_link = out_pred[:,:,:,1]
    gt_text = gt_pred[:,:,:,0]
    gt_link = gt_pred[:,:,:,1]
        
    return torch.mean(torch.abs(out_text - gt_text))# + torch.abs(out_link - gt_link))

def SSIM(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    return ssim(out_image, gt_image, data_range=1, size_average=True)

def PSNR(out_image, gt_image):
    out_image = torch.clamp(out_image, min=0.0, max=1.0)
    mse = torch.mean((out_image - gt_image) ** 2)
    return 10 * torch.log10(1.0 / mse)

# def ColorConstancyLoss(out_image, gt_image):
#     # Gray World Assumption
#     BS, C, H, W = out_image.shape
#     # for bs in range(BS):
#     # out_img = out_image[bs,:,:,:]
#     # gt_img = gt_image[bs,:,:,:]

#     out_img_R_mean = torch.mean(out_image[:,0,:,:])
#     out_img_G_mean = torch.mean(out_image[:,1,:,:])
#     out_img_B_mean = torch.mean(out_image[:,2,:,:])
#     out_img_Gray_mean = (out_img_R_mean + out_img_G_mean + out_img_B_mean) / 3
#     # out_img_kr = out_img_R_mean / out_img_Gray_mean
#     # out_img_kg = out_img_G_mean / out_img_Gray_mean
#     # out_img_kb = out_img_B_mean / out_img_Gray_mean

#     # print(out_img_R_mean, out_img_G_mean, out_img_B_mean, out_img_Gray_mean, out_img_kr, out_img_kg, out_img_kb)

#     gt_img_R_mean = torch.mean(gt_image[:,0,:,:])
#     gt_img_G_mean = torch.mean(gt_image[:,1,:,:])
#     gt_img_B_mean = torch.mean(gt_image[:,2,:,:])
#     gt_img_Gray_mean = (gt_img_R_mean + gt_img_G_mean + gt_img_B_mean) / 3
#     # gt_img_kr = gt_img_R_mean / gt_img_Gray_mean
#     # gt_img_kg = gt_img_G_mean / gt_img_Gray_mean
#     # gt_img_kb = gt_img_B_mean / gt_img_Gray_mean

#     # print(gt_img_R_mean, gt_img_G_mean, gt_img_B_mean, gt_img_Gray_mean, gt_img_kr, gt_img_kg, gt_img_kb)

#     return torch.abs(out_img_Gray_mean - gt_img_Gray_mean)
