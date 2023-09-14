# Import necessary libraries
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

# Define a custom dataset class named 'HateClipDataset' that inherits from PyTorch's 'Dataset' class
class HateClipDataset(Dataset):
    def __init__(self, hate_data_paths, transform):
        # Constructor method that initializes the dataset
        # 'hate_data_paths': Dictionary containing data file paths
        # 'transform': A data transformation function
        self.hate_data_paths = hate_data_paths  # Store data paths
        self.transform = transform  # Store the data transformation function

    def __len__(self):
        # Special method that returns the length of the dataset
        return len(self.hate_data_paths)

    def load_img_pil(self, image_path):
        # Helper function to load an image using the PIL library
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # Convert image to RGB format

    def load_queries(self, hate_meme_caption_item, hate_meme_image_item):
        # Helper function to load queries (captions and images)
        caption = hate_meme_caption_item  # Get the caption
        image_path = hate_meme_image_item  # Get the image path
        caption_tokenized = clip.tokenize(caption, truncate=True)  # Tokenize the caption using CLIP
        pil_img = self.load_img_pil(image_path)  # Load and convert the image to PIL format
        transform_img = self.transform(pil_img)  # Apply the specified data transformation to the image
        return transform_img, caption_tokenized  # Return the transformed image and caption

    def __getitem__(self, idx):
        # Special method to retrieve an item from the dataset
        if torch.is_tensor(idx):
            idx = idx.tolist()  # Convert 'idx' to a Python list if it's a tensor

        # Determine the label based on whether the item is labeled as hate (1) or not (0)
        label = torch.as_tensor(1) if self.hate_data_paths[str(idx)]['label'] else torch.as_tensor(0)

        # Extract caption and image paths from the dataset
        hate_meme_caption_item = self.hate_data_paths[str(idx)]["text"]
        hate_meme_image_item = self.hate_data_paths[str(idx)]["img"]

        # Load and process the queries (caption and image)
        qImg, qCap = self.load_queries(hate_meme_caption_item, hate_meme_image_item)

        # Return the label, transformed image, and caption
        return label, qImg, qCap
