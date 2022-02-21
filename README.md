# Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation
This is my Final Year Project which works on carry out low light image enhancement for better scene text detection, click [here](https://drive.google.com/drive/folders/1u1ACc2EpO4phkqlqkoJ9HUKGuGgNEx-5?usp=sharing) for the slides and video explaining on this work in details.

The overall concept of the work is to carry out a low light enhancement specifically on the text regions of low light input img and get the enhanced img, where later on, when a text detector is applied on the enhanced img,  the text regions in the enhanced img can be clearly detected.
![concept](https://drive.google.com/file/d/1NF5XW9ZqM6WodFWhLB4FYvZIbjUwbkVs/view?usp=sharing)

## Research Objectives
1)To investigate and introduce feasible framework for low light image enhancement that restore image’s text region details using:
- segmentation approach 
 -attention 
 -text detection approach
and have comparable performance with existing baseline model.

2)Introduce a text (segmentation/detection) loss to make the object(text) in enhanced image to be visual clearly.

Novelty : Enhance the text regions of low light image, where in our proposed method text regions will be focused and enhanced together with the background region as a whole.

## Network Architecture
In this model, the core enhancement network,2-heads UNet is guided by (1) 2 attention module (triplet attention and coordinate attention) through multiplication of feature map with attention map, and (2) a text detector.
The backbone of this core enhancement network is based on UNet, which is a fully connected network that combine features from different spatial regions of the image.This architecture can capture the fine grain features of the image,can localize more precisely the regions of interests of the image and is able to train end-to-end.
The attention mechanism helps to focus on enhancing the ROI (text regions), the text detection helps to localize characters in the image, by computing the region score (the probability of the characters) between enhanced image and groundtruth image and the text segmentation helps to guide the learning on text regions by segment out the text regions, get a mask out on the text regions in the image and compare it with the ground truth segmentation image.

![Architecture](https://drive.google.com/file/d/1GByeJxFEpJhSSvGWJwaWROBuP3m-eSUr/view?usp=sharing)

## Requirements
- RawPy 0.13.1
- SciPy 1.0.0
```
pip install -r requirements.txt
```
## Platform
- ubuntu 20.04
- nvidia TITAN Xp gpu
- cuda 11.4
- python 3.9.7
- pytorch 1.8.0

## Dataset
The dataset used in this is Incidental Scene Text(ICDAR15) dataset [here](https://rrc.cvc.uab.es/?ch=4&com=downloads), which I already downloaded and placed it in "dataset" folder.

To simulate the low light scenario, I reduce the brightness of the images to 0.04 times of its original brightness to make it visually similar to low light image dataset, to form the paired images which are required for training and testing.

The script to create text segmentation groundtruth and to adjust brightness of original ICDAR15 image to simulate low light dataset can be get in 'create_binary_textseg_map.ipynb'.

## UI
I build a simple user interface using Streamlit to demonstrate the work (only can run on local pc). User can select an low light input image, then the enhanced image and the text detection of the enhanced image(show in bbx) will be returned.

To run the streamlit app:
```
streamlit run main.py
```

## Testing
The pretrained model [here](https://github.com/meiyihTan/Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation/blob/master/IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/early_stop_model.pth) is in `./Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation/IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss`. 

To test ICDAR15 test data, run
```
python test_IC15_gray_map_TA_CA_msssim_text_detection.py
```
By default, the result will be saved in "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/final_result_pillow" and "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/final_result_torchvision"  folder.

## Training
To train the model, run
```
python train_IC15_gray_seg_TA_CA_msssim_text_detection.py
```
The weight,result and model will be saved in "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss" folder by default.

###### Several different combination of attention modules(CBAM,TA,CA,SP,...) and type of inputs(with edge or seg) were experimented in "train_on_other_combination" folder and can be trained on more possible combination by changing the UNet in unet_seg.py or unet.py.
