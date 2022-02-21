from __future__ import division
import os, time, scipy.io, scipy.misc
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
import numpy as np
import glob
from PIL import Image
import utils
from unet import GrayEdgeAttentionUNet
from dataset.ICDAR15 import ICDAR15Dataset

from CRAFTpytorch.craft import CRAFT
from collections import OrderedDict

input_dir = './dataset/IC15_004/train/low/'
gt_dir = './dataset/IC15_004/train/high/'
list_file= './dataset/IC15_004/train_list.txt'
checkpoint_dir = './IC15_004_results/result_IC15__msssim_text_detection/'
result_dir = checkpoint_dir

w1, w2, w3 = 0.85, 0.15, 0.85*0.55

writer = SummaryWriter(log_dir=checkpoint_dir + 'logs')

bs = 8
ps = 512  # patch size for training
save_freq = 10

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = GrayEdgeAttentionUNet()
unet.to(device)

learning_rate = 1e-4
G_opt = optim.Adam(unet.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.MultiStepLR(G_opt, milestones=[2000], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_opt, factor=0.1, patience=10, verbose=True)
early_stopping = EarlyStopping(patience=30, verbose=True)

dataset = ICDAR15Dataset(list_file = list_file, root_dir = './dataset/', ps=ps)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
iteration = 0


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
        continue
    g_loss = []
    cnt = 0

    # Training
    unet.train()
    for sample in iter(dataloader):
        st = time.time()
        cnt += bs

        in_imgs = sample['in_img'].to(device)
        gt_imgs = sample['gt_img'].to(device)
        in_gray_imgs = sample['in_gray_img'].to(device)
        in_edge_imgs = sample['in_edge_img'].to(device)

        
        torchvision.utils.save_image(in_imgs , f"./in_imgs.png")    
        torchvision.utils.save_image(gt_imgs , f"./gt_imgs.png")    
        torchvision.utils.save_image(in_gray_imgs , f"./in_gray_imgs.png")    
        torchvision.utils.save_image(in_edge_imgs , f"./in_edge_imgs.png")
                        
                
                
        G_opt.zero_grad()
        out_imgs = unet(in_imgs, in_gray_imgs, in_edge_imgs)

        mae_loss = utils.MAELoss(out_imgs, gt_imgs)
        ms_ssim_loss = utils.MS_SSIMLoss(out_imgs, gt_imgs)
        text_loss = utils.TextDetectionLoss(out_imgs, gt_imgs, craft_net)
        loss = w1*mae_loss + w2*ms_ssim_loss + w3*text_loss
        loss.backward()
        G_opt.step()

        g_loss.append(loss.item())

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss), time.time() - st))

        outputs = out_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        outputs = np.minimum(np.maximum(outputs,0),1)
        in_imgs = in_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        in_imgs = np.minimum(np.maximum(in_imgs,0),1)
        gt_imgs = gt_imgs.permute(0, 2, 3, 1).cpu().data.numpy()
        gt_imgs = np.minimum(np.maximum(gt_imgs,0),1)

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((in_imgs[0, :, :, :], gt_imgs[0, :, :, :], outputs[0, :, :, :]), axis=1)
            Image.fromarray((temp * 255).astype('uint8')).convert('RGB').save(result_dir + '%04d/%05d_train.jpg' % (epoch, sample['ind'][0]))  
            torch.save(unet.state_dict(), checkpoint_dir + str(epoch)+'save_mix_best_model.pth')
            print('save new mix best model.')

        iteration += 1
        writer.add_scalar('Train/MAE_Loss', mae_loss, iteration)
        writer.add_scalar('Train/G_Loss', loss, iteration)
        writer.add_scalar('Train/G_Loss_Mean', np.mean(g_loss), iteration)
    
    #end of one epoch
    mean_loss = np.mean(g_loss) 
    early_stopping(mean_loss, unet)
    

    if early_stopping.early_stop:
        print("Early stopping")
        torch.save(unet.state_dict(), checkpoint_dir + 'early_stop_model.pth')
        break #early stopping applied
            	
            	
    
    
    scheduler.step(mean_loss)
    torch.save(unet.state_dict(), checkpoint_dir + 'final_model.pth')
    writer.close()
    
