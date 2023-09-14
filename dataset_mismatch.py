import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import json
from urllib.parse import urlparse
from PIL import Image
import os 
import clip 
import pickle 

class HateClipDataset(Dataset):
    def __init__(self, hate_data_paths, transform):

        self.hate_data_paths = hate_data_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.hate_data_paths)   

    def load_img_pil(self,image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
                       
    def load_queries(self,hate_meme_caption_item, hate_meme_image_item):
        # caption = hate_meme_caption_item['caption']
        caption = hate_meme_caption_item
        image_path = hate_meme_image_item
        caption_tokenized = clip.tokenize(caption,truncate=True) 
        # image_path = os.path.join(self.hate_meme_root_dir, hate_meme_image_item['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption_tokenized

    def __getitem__(self, idx):      
        if torch.is_tensor(idx):
            idx = idx.tolist()    
        
        label = torch.as_tensor(1) if self.hate_data_paths[str(idx)]['label'] else torch.as_tensor(0)
        
        hate_meme_caption_item = self.hate_data_paths[str(idx)]["text"]
        hate_meme_image_item = self.hate_data_paths[str(idx)]["img"]

        qImg, qCap = self.load_queries(hate_meme_caption_item, hate_meme_image_item)

        return label, qImg, qCap