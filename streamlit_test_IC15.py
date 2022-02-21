from __future__ import division
import os, scipy.io, scipy.misc, cv2
import torch
import numpy as np
import glob
import utils
from PIL import Image

from unet_seg import gray_TA_CA_seg_UNet
from torch.utils.data import DataLoader
from dataset.ICDAR15_seg import ICDAR15TestDataset

class ICDAR15Test_streamlit():
    def __init__(self, image_path):
        self.image_path = image_path
        
    def get_own_dataset_output(self, image_path):    
        input_dir = './dataset/IC15_004/test/low/'#'IC_15_test_dataset_streamlit/test_data_low'
        gt_dir = './dataset/IC15_004/test/high/'#'IC_15_test_dataset_streamlit/test_data_high'
        test_list_file= './dataset/IC15_004/test_list.txt'#'IC_15_test_dataset_streamlit/test_list.txt'
        checkpoint_dir =  './IC15_004_results/result_IC15_baseline_gray_seg_TA_CA_msssim_text_detection/'#'IC_15_test_dataset_streamlit/'
        result_dir = checkpoint_dir
        ckpt = checkpoint_dir + 'final_model.pth' #change to model1.pth if get from the drive 


        # get test IDs
        test_fns = glob.glob(gt_dir + '*.jpg')
        test_ids = [int(((os.path.basename(test_fn).split('/')[-1]).split('.')[0])) for test_fn in test_fns]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        unet = gray_TA_CA_seg_UNet()
        unet.load_state_dict(torch.load(ckpt,map_location ='cpu'))
        unet.to(device)

        test_dataset = ICDAR15TestDataset(list_file=test_list_file, root_dir='./dataset/')
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        iteration = 0

        with torch.no_grad():
            unet.eval()
            psnr, ssim = [], []

            for sample in iter(test_dataloader):
                
                in_fn = sample['in_fn'][0]
                print(in_fn)
                if in_fn==(self.image_path.split('/')[-1]):
                    in_img = sample['in_img'].to(device)
                    gt_img = sample['gt_img'].to(device)
                    in_gray_img = sample['in_gray_img'].to(device)
                    
                    out_img,seg_out= unet(in_img,in_gray_img)

                    psnr.append(utils.PSNR(out_img, gt_img).item())
                    ssim.append(utils.SSIM(out_img, gt_img).item())

                    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                    output = np.minimum(np.maximum(output, 0), 1)
                    output = output[0, :, :, :]
                    
                    out_img=Image.fromarray((output * 255).astype('uint8')).convert('RGB')#.save(output_dir+'out.png')
                    
                    #do craft 
                    #do_craft(epoch)# bbx txt saved at  -->./result/+ str(epoch)+ '_halfway_pred/'


                    print('PSNR=%.2f SSIM=%.3f' % (np.mean(psnr), np.mean(ssim)))
                    return out_img,psnr,ssim

