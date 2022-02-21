# Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation
This is my Final Year Project which works on carry out low light image enhancement for better scene text detection, click [here](https://drive.google.com/drive/folders/1u1ACc2EpO4phkqlqkoJ9HUKGuGgNEx-5?usp=sharing) for the slides and video explaining on this work in details.

The **overall concept** of the work is to carry out a low light enhancement specifically on the text regions of low light input image and get the enhanced image, where later on, when a text detector is applied on the enhanced image,  the text regions in the enhanced image can be clearly detected.

![concept](https://drive.google.com/uc?export=view&id=1jiA_kUUUiJADQIkdwkBNxxkLn3GQ_ENK)

## Applications
![application]((https://drive.google.com/uc?export=view&id=1Rl_8PSyFaUK1m1bBP-z-YgbndqKcybZf)

## Research Objectives
1)To investigate and introduce feasible framework for low light image enhancement that restore image’s text region details using:
- segmentation approach 
- attention 
- text detection approach
and have comparable performance with existing baseline model.

2)Introduce a text (segmentation/detection) loss to make the object(text) in enhanced image to be visual clearly.

*Novelty : Enhance the text regions of low light image, where in our proposed method text regions will be focused and enhanced together with the background region as a whole.*

## Network Architecture
In this model, the core enhancement network, 2-heads UNet is guided by **(1) 2 attention module (triplet attention and coordinate attention) through multiplication of feature map with attention map, and (2) a text detector.**

![Architecture](https://drive.google.com/uc?export=view&id=1op6WsaFJmnedZTRj1-bSk2aUwl_Vy_jw)

The **backbone** of this core enhancement network is based on **UNet**, which is a fully connected network that combine features from different spatial regions of the image.This architecture can capture the fine grain features of the image,can localize more precisely the regions of interests of the image and is able to train end-to-end.

The **attention mechanism** helps to **focus on enhancing the ROI (text regions)**, the **text detection** helps to **localize characters in the image**, by computing the region score (the probability of the characters) between enhanced image and groundtruth image and the **text segmentation** helps to **guide the learning on text regions** by segment out the text regions, get a mask out on the text regions in the image and compare it with the ground truth segmentation image.

## Requirements
- RawPy 0.13.1
- SciPy 1.0.0
```
pip install -r requirements.txt
```
Install other prerequisite packages:
```shell
pip install shapely Polygon3
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
I build a simple user interface using Streamlit to demonstrate the work (only can run on local pc). User can select an low light input image, then the enhanced image and the text detection of the enhanced image(show in bounding box) will be returned.

To run the streamlit app:
```
streamlit run main.py
```

## Testing
The [pretrained model ](https://github.com/meiyihTan/Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation/blob/master/IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/early_stop_model.pth) is in `./Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation/IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss`. 

To test ICDAR15 test data, run
```
python test_IC15_gray_map_TA_CA_msssim_text_detection.py
```
By default, the result will be saved in :
- "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/final_result_pillow" 
- "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss/final_result_torchvision" 

## Training
To train the model, run 
```
python train_IC15_gray_seg_TA_CA_msssim_text_detection.py
```
The weight, result and model will be saved in "IC15_004_results/result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss"  folder by default.

## Performance Testing
1)To get on the psnr and ssim value of the inference output image, run
```
python icdar_compute_psnr_ssim.py
```
Change the directory in line 14 to the directory where you save the testing results.

2)To get on the IoU, SIoU and TIoU score of the inference output image,
- download [craft_ic15_20k.pth](https://drive.google.com/file/d/1J552AE1uG0d1ew4ubLI_PjkaOhPdUCSx/view?usp=sharing) and place it into 'CRAFT-pytorch' folder.
- in 'CRAFT-pytorch' folder, run
```
python test.py --trained_model=craft_ic15_20k.pth --test_folder=result_IC15_baseline_gray_fullTextSeg_TA_CA_msssim_text_detection_text_det_loss
```
*test_folder=[folder path to test images]*

The result image and socre maps will be saved to 'CRAFT-pytorch/result' folder by default.
- move all the '.txt' file in the 'CRAFT-pytorch/result' folder to 'CRAFT-pytorch/TIoU-CC/results' folder.
- rename the '.txt' file to "res_img_([0-9]+).txt", where 0-9 is the test image id.
- Change the folder name in line 45 of main.py in 'TIoU-CC' folder to the folder you newly saved all the moved and renamed '.txt' file.
- in 'TIoU-CC' folder,  run
```
python main.py 
```
## 
*Several different combination of attention modules(CBAM,TA,CA,SP,...) and type of inputs(with edge or seg) were experimented in "train_on_other_combination" folder. Move the train....py file out to parent folder,'Cycle-Text-Consistency-for-Image-to-Image-Domain-Adaptation' folder, if training want to be carry out. One can also try out on more possible combination by changing the UNet in unet_seg.py or unet.py.*
