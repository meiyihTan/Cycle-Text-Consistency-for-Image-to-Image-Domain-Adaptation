from __future__ import division
import os, scipy.io, scipy.misc, cv2
import numpy as np
import glob
import math
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

gt_dir = '/media/meiyih/meiyih_datasets/baseline_UNet/dataset/IC15_004/test/test_high/'

psnrs, ssims = [], []
cnt=0
print('can load')
for file in glob.glob('/media/meiyih/meiyih_datasets/baseline_UNet/IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/final_result_pillow/'+"*.jpg"):
    in_fn = file
    print(in_fn)
    fn_idx = in_fn.split('/')[-1]
    gt_fn = gt_dir + str(fn_idx)

    in_img = cv2.imread(in_fn)
    gt_img = cv2.imread(gt_fn)
    cnt+=1

    psnrs.append(peak_signal_noise_ratio(gt_img, in_img))
    ssims.append(structural_similarity(gt_img, in_img, multichannel=True))

print('np.mean(psnrs) : ',np.mean(psnrs))
print('np.mean(ssims) : ',np.mean(ssims))
print('cnt is',cnt)

