import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

import matplotlib.pyplot as plt

class ICDAR15Dataset_seg(Dataset):
    """ICDAR15 dataset."""
    def __init__(self, list_file ,root_dir, ps,transform=None):
        self.ps = ps
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        #self.gt_images = [None] * 1001
        #self.input_images = [None] * 1001
        self.input_gray_images = [None] * 1001
        #self.input_edge_images = [None] * 1001
        #self.input_seg_images = [None] * 1001

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        gt_img_name = gt_img_name.split('\n')[0]

        ratio = 1
        ind = input_img_name.split('/')[-1]
        ind = ind.split('.')[0]
        ind = int(ind)

        #if self.input_images[ind] is None:
        input_img_path = os.path.join(self.root_dir, input_img_name)
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
        #self.input_images[ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio

        gt_img_path = os.path.join(self.root_dir, gt_img_name)
        im = cv2.imread(gt_img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.expand_dims(np.float32(im / 255.0), axis=0)
            #self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)

            # gray_path = os.path.join(self.root_dir, 'IC15_004/train/gray/%d.png' % (ind))
            # input_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            # self.input_gray_images[ind] = np.expand_dims(np.expand_dims(np.float32(input_gray / 255.0), axis=2), axis=0)

        seg_path = os.path.join(self.root_dir, 'IC15_004/train/masks/%d.jpg' % (ind))# old_no_full_text_masks
        input_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        _, input_seg = cv2.threshold(input_seg , 127, 255, cv2.THRESH_BINARY)
        input_seg = np.expand_dims(np.expand_dims(np.float32(input_seg / 255.0), axis=2), axis=0)
            
        # crop
        H = input_img.shape[1]
        W = input_img.shape[2]
        print(H,W)
        dim1=720
        dim2 =1280
        assert input_img.shape[1] == dim1
        assert input_img.shape[2] == dim2

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = input_img[:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = im[:, yy:yy + self.ps, xx:xx + self.ps, :]
        # input_gray_patch = self.input_gray_images[ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        input_seg_patch = input_seg[:, yy:yy + self.ps, xx:xx + self.ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            # input_gray_patch = np.flip(input_gray_patch, axis=1)
            input_seg_patch = np.flip(input_seg_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
            # input_gray_patch = np.flip(input_gray_patch, axis=2)
            input_seg_patch = np.flip(input_seg_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
            # input_gray_patch = np.transpose(input_gray_patch, (0, 2, 1, 3))
            input_seg_patch = np.transpose(input_seg_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # input_gray_patch = np.maximum(input_gray_patch, 0.0)
        input_seg_patch = np.maximum(input_seg_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_patch).permute(0,3,1,2)
        in_seg_img = torch.from_numpy(input_seg_patch).permute(0,3,1,2)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)
        
        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_seg_img': in_seg_img.squeeze(0), 'ind': ind, 'ratio': ratio}

        return sample



class ICDAR15Dataset_seg_alpha(Dataset):
    """ICDAR15 dataset."""
    def __init__(self, list_file ,root_dir, ps,transform=None):
        self.ps = ps
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        #self.gt_images = [None] * 1001
        #self.input_images = [None] * 1001
        #self.input_gray_images = [None] * 1001
        #self.input_edge_images = [None] * 1001
        #self.input_seg_images = [None] * 1001

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        gt_img_name = gt_img_name.split('\n')[0]

        ratio = 1
        ind = input_img_name.split('/')[-1]
        ind = ind.split('.')[0]
        ind = int(ind)

        #if self.input_images[ind] is None:
        input_img_path = os.path.join(self.root_dir, input_img_name)
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
        #self.input_images[ind] = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio

        gt_img_path = os.path.join(self.root_dir, gt_img_name)
        im = cv2.imread(gt_img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.expand_dims(np.float32(im / 255.0), axis=0)
            #self.gt_images[ind] = np.expand_dims(np.float32(im / 255.0), axis=0)
	
	#the alpha channel to be added to the multiplied img # read mask as binary image (1 channel)        
        alpha_path = os.path.join(self.root_dir, 'IC15_004/train/masks/%d.jpg' % (ind))
        input_alpha = cv2.imread(alpha_path,cv2.IMREAD_UNCHANGED)
        _, alpha_seg = cv2.threshold(input_alpha , 127, 255, cv2.THRESH_BINARY)
        alpha_seg[alpha_seg == 0] = 153
        #alpha_seg = np.expand_dims(np.float32(alpha_seg/ 255.0), axis=0)
        #alpha_seg = np.float32(alpha_seg/ 255.0)
        alpha_seg = np.expand_dims(np.expand_dims(np.float32(alpha_seg/ 255.0), axis=2), axis=0)    
        #print('alpha_seg.shape : ',alpha_seg.shape)    
        
    
	#the segmask mask for multiply  # read mask as rgb image (will straight have 3 channel)
        mul_seg_path = os.path.join(self.root_dir, 'IC15_004/train/masks/%d.jpg' % (ind))
        input_mul_seg = Image.open(mul_seg_path)
        input_mul_seg = input_mul_seg.convert('RGB')    
        input_mul_seg=np.array(input_mul_seg)    
        _, input_mul_seg = cv2.threshold(input_mul_seg , 127, 255, cv2.THRESH_BINARY)
        input_mul_seg =  np.expand_dims(np.float32(input_mul_seg / 255.0), axis=0)
	
	#seg mask for segmentation head 
        seg_path = os.path.join(self.root_dir, 'IC15_004/train/masks/%d.jpg' % (ind))
        input_seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        _, input_seg = cv2.threshold(input_seg , 127, 255, cv2.THRESH_BINARY)
        input_seg = np.expand_dims(np.expand_dims(np.float32(input_seg / 255.0), axis=2), axis=0)
            
        # crop
        H = input_img.shape[1]
        W = input_img.shape[2]

        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = input_img[:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = im[:, yy:yy + self.ps, xx:xx + self.ps, :]
        # input_gray_patch = self.input_gray_images[ind][:, yy:yy + self.ps, xx:xx + self.ps, :]
        alpha_seg_patch = alpha_seg[:, yy:yy + self.ps, xx:xx + self.ps, :]
        input_mul_seg_patch = input_mul_seg[:, yy:yy + self.ps, xx:xx + self.ps, :]
        input_seg_patch = input_seg[:, yy:yy + self.ps, xx:xx + self.ps, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            # input_gray_patch = np.flip(input_gray_patch, axis=1)
            alpha_seg_patch  = np.flip(alpha_seg_patch, axis=1)
            input_mul_seg_patch = np.flip(input_mul_seg_patch, axis=1)
            input_seg_patch = np.flip(input_seg_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
            # input_gray_patch = np.flip(input_gray_patch, axis=2)
            alpha_seg_patch  = np.flip(alpha_seg_patch, axis=2)
            input_mul_seg_patch = np.flip(input_mul_seg_patch, axis=2)               
            input_seg_patch = np.flip(input_seg_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
            # input_gray_patch = np.transpose(input_gray_patch, (0, 2, 1, 3))
            alpha_seg_patch = np.transpose(alpha_seg_patch, (0, 2, 1, 3))
            input_mul_seg_patch = np.transpose(input_mul_seg_patch, (0, 2, 1, 3))
            input_seg_patch = np.transpose(input_seg_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # input_gray_patch = np.maximum(input_gray_patch, 0.0)
        alpha_seg_patch = np.maximum(alpha_seg_patch, 0.0)
        input_mul_seg_patch= np.maximum(input_mul_seg_patch, 0.0)
        input_seg_patch = np.maximum(input_seg_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_patch).permute(0,3,1,2)
        alpha_seg_img = torch.from_numpy(alpha_seg_patch).permute(0,3,1,2)
        input_mul_seg_img = torch.from_numpy(input_mul_seg_patch).permute(0,3,1,2)
        in_seg_img = torch.from_numpy(input_seg_patch).permute(0,3,1,2)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'in_seg_img': in_seg_img.squeeze(0),'alpha_seg_img': alpha_seg_img.squeeze(0),'input_mul_seg_img': input_mul_seg_img.squeeze(0), 'ind': ind, 'ratio': ratio}

        return sample

class ICDAR15TestDataset(Dataset):
    """ICDAR15 Test dataset."""
    def __init__(self, list_file ,root_dir):
        self.list_file = open(list_file, "r")
        self.list_file_lines = self.list_file.readlines()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.list_file_lines)

    def __getitem__(self, idx):
        img_names = self.list_file_lines[idx].split(' ')
        input_img_name = img_names[0]
        gt_img_name = img_names[1]
        gt_img_name = gt_img_name.split('\n')[0]

        ratio = 1
        ind = input_img_name.split('/')[-1]
        ind = ind.split('.')[0]
        ind = int(ind)
        
        in_fn = input_img_name.split('/')[-1]
        rm_input_img_name = input_img_name[2:] #remove ./ in ./IC15_004/test/low/1.jpg

        input_img_path = os.path.join(self.root_dir, rm_input_img_name)
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_image_full = np.expand_dims(np.float32(input_img / 255.0), axis=0) * ratio
        
        rm_gt_img_name = gt_img_name[2:] #remove ./ in ./IC15_004/test/high/1.jpg
        gt_img_path = os.path.join(self.root_dir, rm_gt_img_name)
        im = cv2.imread(gt_img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gt_image_full = np.expand_dims(np.float32(im / 255.0), axis=0)

        # gray_path = os.path.join(self.root_dir, 'ICDAR15/test/gray/%d.png' % (ind))
        # input_gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        # input_gray_image_full = np.expand_dims(np.expand_dims(np.float32(input_gray / 255.0), axis=2), axis=0)

        # edge_path = os.path.join(self.root_dir, 'ICDAR15/test/edge/%d.png' % (ind))
        # input_edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        # input_edge_image_full = np.expand_dims(np.expand_dims(np.float32(input_edge / 255.0), axis=2), axis=0)

        input_image_full = np.minimum(input_image_full, 1.0)
        gt_image_full = np.maximum(gt_image_full, 0.0)
        # input_gray_image_full = np.maximum(input_gray_image_full, 0.0)
        # input_edge_image_full = np.maximum(input_edge_image_full, 0.0)
        
        in_img = torch.from_numpy(input_image_full).permute(0,3,1,2)
        gt_img = torch.from_numpy(gt_image_full).permute(0,3,1,2)
        # in_gray_img = torch.from_numpy(input_gray_image_full).permute(0,3,1,2)
        # in_edge_img = torch.from_numpy(input_edge_image_full).permute(0,3,1,2)

        r,g,b = in_img[0,0,:,:]+1, in_img[0,1,:,:]+1, in_img[0,2,:,:]+1
        in_gray_img = (1.0 - (0.299*r+0.587*g+0.114*b)/2.0).unsqueeze(0).unsqueeze(0)

        sample = {'in_img': in_img.squeeze(0), 'gt_img': gt_img.squeeze(0), 'in_gray_img': in_gray_img.squeeze(0), 'ind': ind, 'ratio': ratio, 'in_fn': in_fn}

        return sample
