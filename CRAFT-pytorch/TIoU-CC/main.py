#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import zipfile

import ic15.rrc_evaluation_funcs as tiou
import ic15.script as tiou_script

import curved_tiou.rrc_evaluation_funcs as ctiou
import curved_tiou.script as ctiou_script

def eval(dataset_name, gt_path, res_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    params = {
        'flag': dataset_name,
        'o': output_path,
        's': res_path,
        'g': gt_path
    }

    if dataset_name == 'icdar15':
        resDict = tiou.main_evaluation(params,tiou_script.default_evaluation_params,tiou_script.validate_data,tiou_script.evaluate_method)
    else:
        resDict = ctiou.main_evaluation(params,ctiou_script.default_evaluation_params,ctiou_script.validate_data,ctiou_script.evaluate_method)
    
    return resDict


if __name__=='__main__':
    root_path = '/media/meiyih/meiyih_datasets/baseline_UNet/CRAFT-pytorch/TIoU-CC/'
    dataset_name = 'icdar15' # icdar15 or total_text
    gt_path = ''
    if dataset_name == 'icdar15':
        gt_path = os.path.join(root_path, 'ic15/gt.zip')
    elif dataset_name == 'total_text':
        gt_path = os.path.join(root_path, 'curved_tiou/totaltext_GT.zip')
    elif dataset_name == 'ctw': 
        gt_path = os.path.join(root_path, 'curved_tiou/ctw_GT.zip')

    run_name = 'TA_SP_CA_torchvision'
    res_path = os.path.join(root_path, 'results/cv2weight_pil')

    txts = os.listdir(res_path)
    zip_path = str(Path(res_path).parents[0])+'/{}_{}.zip'.format(dataset_name, run_name)
    with zipfile.ZipFile(zip_path, 'a') as zip:
        for txt_name in txts:
            zip.write(filename = os.path.join(res_path, txt_name), arcname = txt_name)

    output_path = os.path.join(root_path, f'eval_output/{dataset_name}/{run_name}')
    resDict = eval(dataset_name, gt_path, zip_path, output_path)
    hmean = round(resDict['method']['hmean'],3)
    print(dataset_name, hmean)
