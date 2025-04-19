import numpy as np
from PIL import Image
import torch
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


## https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
class peligemmaProcessor:
    IMAGE_TOKEN= "<image>"

    def __init__(self,tokenizer,num_token_size,image_size):
        self.num_token_size=num_token_size
        self.image_size=image_size
        self.tokenizer = tokenizer

        tokens_to_add={"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]  
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]  
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
    
    def __call__(self, text , images , padding ="longest", truncation=True):
        pixels_value=process_images(images,size=(self.image_size,self.image_size),resample=Image.Resampling.BICUBIC,rescale_factor=1 / 255.0,image_mean=IMAGENET_STANDARD_MEAN,image_std=IMAGENET_STANDARD_STD)
        pixels_value=np.stack(pixels_value,axis=0)
        pixels_value=torch.tensor(pixels_value, dtype=torch.float32)


        input_strings = [add_image_tokens_to_prompt(prefix_prompt=prompt,bos_token=self.tokenizer.bos_token,image_seq_len=self.num_token_size,image_token=self.IMAGE_TOKEN)
            for prompt in text]
        
        inputs = self.tokenizer(input_strings,return_tensors="pt",padding=padding,truncation=truncation)

        return_data = {"pixel_values": pixels_value, **inputs}

        return return_data

def process_images(images,size,resample,rescale_factor,image_mean,image_std):
    height , width=size[0], size[1]
    images=[resize(img,size=(height,width),resample=resample) for img in images]
    images=[np.array(img) for img in images]
    images=[rescale(img,scale=rescale_factor) for img in images]
    images=[normalize(img,mean=image_mean,std=image_std) for img in images]
    images=[img.transpose(2,0,1) for img in images]
    return images

def resize(img,size,resample,reducing_gap=None):
    height,width=size
    resized_image=img.resize( (width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image

def rescale(img,scale,dtype= np.float32):
    rescaled_image = img * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(img,mean,std):
    mean = np.array(mean, dtype=img.dtype)
    std = np.array(std, dtype=img.dtype)
    img = (img - mean) / std
    return img

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"




       
