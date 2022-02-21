from __future__ import division
import os, time, scipy.io, scipy.misc
from tqdm import tqdm
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
import numpy as np
import glob
import cv2
from PIL import Image
import utils
from unet_seg import gray_TA_CA_seg_UNet
from dataset.ICDAR15_seg import ICDAR15Dataset_seg_alpha
import torchvision
from CRAFTpytorch.craft import CRAFT
from collections import OrderedDict

input_dir = './dataset/IC15_004/train/low/'
gt_dir = './dataset/IC15_004/train/high/'
list_file= './dataset/IC15_004/train_list.txt'
checkpoint_dir = './IC15_004_results/train_IC15_gray_fullTextSeg_TA_CA_msssim_cv2addweightLoss_text_detection/'
result_dir = checkpoint_dir



bs = 2
ps = 512  # patch size for training
save_freq = 10

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = gray_TA_CA_seg_UNet()
unet.to(device)

learning_rate = 1e-4
G_opt = optim.Adam(unet.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.MultiStepLR(G_opt, milestones=[2000], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_opt, factor=0.1, patience=10, verbose=True)
early_stopping = EarlyStopping(patience=20, verbose=True)

dataset = ICDAR15Dataset_seg_alpha(list_file = list_file, root_dir = './dataset/', ps=ps)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
iteration = 0
BCEloss_fn = nn.BCEWithLogitsLoss()


# text detection model
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
# CRAFT net
craft_net = CRAFT()
craft_pretrained_model = './CRAFTpytorch/craft_ic15_20k.pth'
craft_net.load_state_dict(copyStateDict(torch.load(craft_pretrained_model)))
craft_net.to(device)
craft_net.eval()

for epoch in tqdm(range(lastepoch, 4001)):
    if os.path.isdir(result_dir + '%04d' % epoch):
        os.makedirs(result_dir + '%04d' % epoch)
        continue
    g_loss = []
    cnt = 0
    # Training
    unet.train()
    
    if not os.path.isdir(result_dir + '%04d' % epoch):
        os.makedirs(result_dir + '%04d' % epoch)

    for sample in iter(dataloader):
        st = time.time()
        cnt += bs

        in_imgs = sample['in_img'].to(device)
        gt_imgs = sample['gt_img'].to(device)
        in_gray_imgs = sample['in_gray_img'].to(device)
        in_seg_imgs = sample['in_seg_img'].to(device)#seg mask/gt
        
        G_opt.zero_grad()
        #out_imgs = unet(in_imgs)
        enh_out,seg_out = unet(in_imgs, in_gray_imgs)      

        mae_loss = utils.MAELoss(enh_out, gt_imgs)
        ms_ssim_loss = utils.MS_SSIMLoss(enh_out, gt_imgs)
        seg_loss = BCEloss_fn(seg_out, in_seg_imgs)
        text_loss = utils.TextDetectionLoss(enh_out, gt_imgs, craft_net)
        
        #print(f'enh_out : {enh_out.shape},{enh_out.max()},{enh_out.min()},{type(enh_out)}')
        #print(f'in_seg_imgs : {in_seg_imgs.shape},{in_seg_imgs.max()},{in_seg_imgs.min()},{type(in_seg_imgs)}')
        #print(f'gt_imgs : {gt_imgs.shape},{gt_imgs.max()},{gt_imgs.min()},{type(gt_imgs)}')
        
        #sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_seg_img': in_seg_img.squeeze(0),'alpha_seg_img': alpha_seg_img.squeeze(0),'input_mul_seg_img': input_mul_seg_img.squeeze(0), 'ind': ind, 'ratio': ratio}
        #overlay seg_img on pred img and gt
        input_mul_seg_imgs = sample['input_mul_seg_img'].to(device)
        #alpha_seg_imgs = sample['alpha_seg_img'].to(device)
        #alpha_seg_imgs =torch.squeeze(torch.squeeze(alpha_seg_imgs))
        
        #print(f'input_mul_seg_imgs : {input_mul_seg_imgs.shape},{input_mul_seg_imgs.max()},{input_mul_seg_imgs.min()},{type(input_mul_seg_imgs)}')
        #print(f'alpha_seg_imgs : {alpha_seg_imgs.shape},{alpha_seg_imgs.max()},{alpha_seg_imgs.min()},{type(alpha_seg_imgs)}')
        
        input_mul_seg_imgs = input_mul_seg_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()#torch.squeeze(input_mul_seg_imgs.permute(0, 2, 3, 1)).cpu().detach().numpy()
        pred_out=enh_out.permute(0, 2, 3, 1).cpu().detach().numpy()
        #alpha_seg_imgs = alpha_seg_imgs.cpu().detach().numpy()
        gt_imgs_for_overlay =gt_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()#torch.squeeze(gt_imgs.permute(0, 2, 3, 1)).cpu().detach().numpy()        
        
        #overlay input_mul_seg_imgs with pred_out    
        result1 = cv2.addWeighted(pred_out, 0.5, input_mul_seg_imgs, 0.5, 0)
        #overlay input_mul_seg_imgs with gt     
        result2 = cv2.addWeighted(gt_imgs_for_overlay, 0.5, input_mul_seg_imgs, 0.5, 0)
        
        
        #mul_pred = np.multiply(enh_out.permute(0, 2, 3, 1).cpu().detach().numpy(), input_mul_seg_imgs)#np.multiply(torch.squeeze(enh_out.permute(0, 2, 3, 1)).cpu().detach().numpy(), input_mul_seg_imgs)
        #print(f'mul_pred : {mul_pred.shape},{mul_pred.max()},{mul_pred.min()},{type(mul_pred)}')
        #rgba_mul_pred = np.dstack((mul_pred,alpha_seg_imgs))
        #print(f'rgba_mul_pred : {rgba_mul_pred.shape},{rgba_mul_pred.max()},{rgba_mul_pred.min()},{type(rgba_mul_pred)}')
        #mul_gt = np.multiply(gt_imgs_for_overlay, input_mul_seg_imgs)
        #print(f'mul_gt : {mul_gt.shape},{mul_gt.max()},{mul_gt.min()},{type(mul_gt)}')
        #rgba_mul_gt = np.dstack((mul_gt ,alpha_seg_imgs))
        #print(f'rgba_mul_gt : {rgba_mul_gt.shape},{rgba_mul_gt.max()},{rgba_mul_gt.min()},{type(rgba_mul_gt)}')
        
        
        #rgba_mul_pred =np.expand_dims(rgba_mul_pred , axis=0)
        #rgba_mul_gt =np.expand_dims(rgba_mul_gt  , axis=0)
        
        
        result1 = torch.from_numpy(result1).permute(0,3,1,2).to(device)  #torch.from_numpy(rgba_mul_pred ).permute(0,3,1,2).to(device) 
        result2 =torch.from_numpy(result2).permute(0,3,1,2).to(device) #torch.from_numpy(rgba_mul_gt).permute(0,3,1,2).to(device)
        #print(f'rgba_mul_pred : {mul_pred.shape},{mul_pred.max()},{mul_pred.min()},{type(mul_pred)}')
        #print(f'rgba_mul_gt : {mul_gt.shape},{mul_gt.max()},{mul_gt.min()},{type(mul_gt)}')        
        
        mae_loss_overlay = utils.MAELoss(result1, result2)
        #outputs_temp = enh_out.permute(0, 2, 3, 1).cpu().data.numpy()
        #outputs_temp = np.minimum(np.maximum(outputs_temp,0),1)        
        #outputs_temp=  outputs_temp[0, :, :, :]
        #Image.fromarray((outputs_temp* 255).astype('uint8')).convert('RGB').save(result_dir + '%04d/%05d_trial_train.jpg' % (epoch, sample['ind'][0]))  
        
        
        loss =2.75*mae_loss+0.25*seg_loss+ 0.075*ms_ssim_loss+0.25*mae_loss_overlay +0.57*text_loss 
        loss.backward()
        G_opt.step()

        g_loss.append(loss.item())

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss), time.time() - st))

        outputs = enh_out.permute(0, 2, 3, 1).cpu().data.numpy()
        outputs = np.minimum(np.maximum(outputs,0),1)
        in_imgs = in_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        in_imgs = np.minimum(np.maximum(in_imgs,0),1)
        gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_imgs = np.minimum(np.maximum(gt_imgs,0),1)
        
        torchvision.utils.save_image(result1 , result_dir + '%04d/%05d_trytryresult1_predout_train.png' % (epoch, sample['ind'][0]))
        torchvision.utils.save_image(result2, result_dir + '%04d/%05d_trytryresult2_gt_train.png' % (epoch, sample['ind'][0]))
        #rgba_mul_pred = rgba_mul_pred.permute(1,2,0).cpu().data.numpy()
        #rgba_mul_pred = np.minimum(np.maximum(rgba_mul_pred,0),1)
        #rgba_mul_gt = rgba_mul_gt.permute(1,2,0).cpu().data.numpy()
        #rgba_mul_gt= np.minimum(np.maximum(rgba_mul_gt,0),1)
       
        #Image.fromarray((rgba_mul_pred * 255).astype('uint8')).convert('RGB').save(result_dir + '%04d/%05d_trytryrgba_mul_pred_train.jpg' % (epoch, sample['ind'][0])) 
        #Image.fromarray((rgba_mul_gt * 255).astype('uint8')).convert('RGB').save(result_dir + '%04d/%05d_trytryrgba_mul_gt_train.jpg' % (epoch, sample['ind'][0])) 



        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((in_imgs[0, :, :, :], gt_imgs[0, :, :, :], outputs[0, :, :, :]), axis=1)
            Image.fromarray((temp * 255).astype('uint8')).convert('RGB').save(result_dir + '%04d/%05d_train.jpg' % (epoch, sample['ind'][0]))  
            torch.save(unet.state_dict(), checkpoint_dir + str(epoch)+'save_mix_best_model.pth')
            print('save new mix best model.')

        iteration += 1
    
    #end of one epoch
    mean_loss = np.mean(g_loss) 
    early_stopping(mean_loss, unet)
    

    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(unet.state_dict(), checkpoint_dir + 'early_stop_model.pth')
        break #early stopping applied
            	
            	
    
    
    scheduler.step(mean_loss)
    torch.save(unet.state_dict(), checkpoint_dir + 'final_model.pth')
 
    
