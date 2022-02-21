import glob
from PIL import Image 
import PIL 

def main():
    print('run')
    for file in glob.glob('/media/meiyih/meiyih_datasets/Pytorch-UNet_run2/rgb_pred/'+"*.png"):
        print(file.split('/')[-1].split('.')[0])
        pil_img = Image.open(file)
        pil_img = pil_img.resize((1280, 720), resample= Image.BICUBIC)
        pil_img.save('/media/meiyih/meiyih_datasets/Pytorch-UNet_run2/resize_rgb_pred/'+(file.split('/')[-1].split('.')[0])+".png")
        
        
        
if __name__ == "__main__":
    main()

