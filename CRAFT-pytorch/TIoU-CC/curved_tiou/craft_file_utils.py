# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc

from shapely.geometry import *

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_file, img)


def saveTexts(img_file, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))
        if 'ctw' not in img_file and 'total_text' not in img_file:
            filename = filename.replace('_fake_B', '')
            if filename.isdigit() and 'img' not in filename:
                filename = 'img_'+filename
                
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # result directory
        if 'ctw' not in img_file and 'total_text' not in img_file:
            res_file = os.path.join(dirname, "res_" + filename + '.txt')
        else:
            res_file = os.path.join(dirname, filename + '.txt')

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                strResult = ''
                poly = np.array(box).astype(np.int32).reshape((-1))
                if 'ctw' not in img_file and 'total_text' not in img_file:
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                else:
#                     print('Processing: ',res_file)
                    cors = poly
                    assert(len(cors) %2 == 0), f'cors length {len(cors)} invalid.'
                    pts = [(int(cors[j]), int(cors[j+1])) for j in range(0,len(cors),2)]
                    try:
                        pgt = Polygon(pts)
                    except Exception as e:
                        print('Not a valid polygon.', pgt)
                        continue

                    if not pgt.is_valid: 
                        print('Polygon has intersecting sides.', pts)
                        continue

                    pRing = LinearRing(pts)
                    if pRing.is_ccw:
                        pts.reverse()
                    outstr= ''
                    for ipt  in pts[:-1]:
                        outstr += (str(int(ipt[0]))+','+ str(int(ipt[1]))+',')
                    outstr += (str(int(pts[-1][0]))+','+ str(int(pts[-1][1])))
                    strResult = outstr+'\n'
                    
                f.write(strResult)


