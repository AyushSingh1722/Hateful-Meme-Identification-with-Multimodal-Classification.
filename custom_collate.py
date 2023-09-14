# Import the PyTorch library
import torch

# Define a function called 'collate_mismatch' that takes a 'batch' of data as input
def collate_mismatch(batch):
    # Extract the labels, images, and captions from the batch
    labels = [item[0] for item in batch]    # Extract labels (item[0]) from each item in the batch
    imgs = [item[1] for item in batch]       # Extract images (item[1]) from each item in the batch

    # Stack the labels and images to create tensors
    labels = torch.stack(labels, dim=0)     # Stack the labels along dimension 0 to create a tensor
    imgs = torch.stack(imgs, dim=0)         # Stack the images along dimension 0 to create a tensor

    # Extract captions from each item in the batch
    captions_batch = [item[2] for item in batch]

    # Concatenate the captions to create a single tensor
    captions_batch = torch.cat(captions_batch, dim=0)

    # Return the labels, images, and captions as a tuple
    return labels, imgs, captions_batch

