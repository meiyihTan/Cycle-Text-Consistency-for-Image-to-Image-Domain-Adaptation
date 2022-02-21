
import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import os
from streamlit_test_IC15 import ICDAR15Test_streamlit
import random
import datetime

def main():
    
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Low Light Image Enhancement')
    )
    
    if selected_box == 'Welcome':
        welcome() 
   
    if selected_box == 'Low Light Image Enhancement':
        low_light_img_enh()
    

def welcome():
    
    st.title('Cycle-Text Consistency for Image-to-Image Domain Adaptation')
    
    st.subheader('Tan Mei Yih 17059516/1' )
    st.subheader(' A simple app that shows low light image enhancement for better scene text detection and several different image processing algorithms. ')
    st.subheader('You can choose the options from the left. I have implemented only a few to show how it works on Streamlit. ' )
    st.image('LLIE_demo_cover_pg.jpg',use_column_width=True)
    st.subheader('Let\'s try this app out!')
    
    


def load_image(filename):
    image = cv2.imread(filename)
    return image
 

def low_light_img_enh():
    a = datetime.datetime.now()
#     uploaded_file = st.file_uploader('Upload an image', type = 'jpg')
    st.header("Low light image enhancement")
    
    image_file_chosen = st.selectbox('Select an existing image:', get_list_of_images())
        
    in_image = Image.open('/media/meiyih/meiyih_datasets/baseline_UNet/dataset/IC15_004/test/test_low/'+image_file_chosen)
    st.image(in_image, caption='Uploaded Image (Input) ', use_column_width=True)
    st.write("Processing Image Enhancement... ")
    out_img,psnr,ssim = predict('/media/meiyih/meiyih_datasets/baseline_UNet/dataset/IC15_004/test/test_low/'+image_file_chosen)
    output_image = out_img#Image.open(out_img)
    st.image(output_image, caption='Enhanced Image (Output) ', use_column_width=True)
    st.write("psnr : ", (psnr[0]))
    st.write("ssim " , (ssim[0]))
    
    #currently this heatmap results are get from pre-inference results from the CRAFT-pytorch/result_TA_CA_torchvision
    #to be added : need to include CRAFT model to do real time inference 
    out_img_text_det = Image.open('/media/meiyih/meiyih_datasets/baseline_UNet/CRAFT-pytorch/result_TA_CA_torchvision/res_'+image_file_chosen)
    st.image(out_img_text_det, caption='Text detection in enhanced Image (Output) ', use_column_width=True)
    out_img_text_heatmaps = Image.open('/media/meiyih/meiyih_datasets/baseline_UNet/CRAFT-pytorch/result_TA_CA_torchvision/res_mask_'+image_file_chosen)
    st.image(out_img_text_heatmaps, caption='Charater heatmap of enhanced Image (Output) ', use_column_width=True)
    #to be added : need to include TIoU-CC to get IoU results 
    #st.write("IoU : ")
    #st.write("SIoU " )   
    #st.write("TIoU ")
    b = datetime.datetime.now()
    st.write("time used : " , ((b-a)))
    

def get_list_of_images():
    file_list = os.listdir('/media/meiyih/meiyih_datasets/baseline_UNet/dataset/IC15_004/test/test_low')
    return [str(filename) for filename in file_list if str(filename).endswith('.jpg')]
    
def predict(image_path): 
    obj=ICDAR15Test_streamlit(image_path)#obj is an instance of ICDAR15Test_streamlit class
    out_img,psnr,ssim=obj.get_own_dataset_output(image_path)
    return out_img,psnr,ssim



    
    
if __name__ == "__main__":
    main()
