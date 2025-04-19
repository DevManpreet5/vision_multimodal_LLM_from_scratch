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
    
    def __call__(self, text , images , padding , truncation):
        pixels_value=process_images(images,size=(self.image_size,self.image_size),resample=Image.Resampling.BICUBIC,rescale_factor=1 / 255.0,image_mean=IMAGENET_STANDARD_MEAN,image_std=IMAGENET_STANDARD_STD)
        pixels_value=np.stack(pixels_value,axis=0)
        pixels_value=torch.tensor(pixels_value)

        input_strings = [add_image_tokens_to_prompt(prefix_prompt=prompt,bos_token=self.tokenizer.bos_token,image_seq_len=self.image_seq_length,image_token=self.IMAGE_TOKEN)
            for prompt in text]
        
        inputs = self.tokenizer(input_strings,return_tensors="pt",padding=padding,truncation=truncation)

        return_data = {"pixel_values": pixels_value, **inputs}

        return return_data

       
