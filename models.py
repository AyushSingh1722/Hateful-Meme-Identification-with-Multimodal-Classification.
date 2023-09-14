# Import necessary libraries and modules
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import BertTokenizer, VisualBertModel
from util import RPN

# Define a class for the Baseline model
class Baseline_model(nn.Module):
    """
    Baseline model for the Hateful Memes classification

    Uses pre-trained vision and text embeddings
    from googlenet (torchvision) and fastText respectively

    Args:
        hidden_size: The size of the hidden layer
        drop_prob: Dropout probability (default is 0.1)
    """
    
    def __init__(self, hidden_size, drop_prob=0.1):
        super(Baseline_model, self).__init__()
        
        # Load a pre-trained vision model (ResNet-152)
        self.vision_pretrain = torchvision.models.resnet152(pretrained=True)

        # Load a pre-trained text embedding model (SentenceTransformer)
        self.text_model = SentenceTransformer("all-mpnet-base-v2")

        # Define linear layers and dropout for classification
        self.fc1 = nn.Linear(1768, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)

    def flatten(self, x):
        # Flatten the input tensor
        N = x.shape[0]
        return x.view(N, -1)

    def forward(self, image, text, device):
        # Pass the image through the vision model
        image = self.vision_pretrain(image)

        # Encode the text using the text embedding model
        text = torch.tensor(self.text_model.encode(text)).squeeze().to(device)

        # Flatten and concatenate the image and text features
        image = self.flatten(F.relu(image))
        text = self.flatten(F.relu(text))
        combined_feat = torch.cat((image, text), dim=1)

        # Pass the combined features through linear layers and apply dropout
        fc1_out = self.fc1(combined_feat)
        fc1_out = self.relu(fc1_out)        
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = self.fc2(fc1_out)

        return fc2_out

# Define a class for the VisualBert model
class VisualBert_Model(nn.Module):
    """
    Model that utilizes VisualBert

    Args:
        batch_size: Batch size
        hidden_size: The size of the hidden layers
        device: The device (CPU or GPU) for computation
        drop_prob: Dropout probability (default is 0.1)
    """
    def __init__(self, batch_size, hidden_size, device, drop_prob=0.1):
        super(VisualBert_Model, self).__init__()
        self.device = device
        self.hidden = hidden_size

        # Load a pre-trained VisualBert model
        self.vbert_model = VisualBertModel.from_pretrained("uclanlp/visualbert-nlvr2-coco-pre")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Initialize the Region Proposal Network (RPN)
        self.RPN = RPN(batch_size, device)
        
        # Define linear layers, layer normalization, and dropout for classification
        self.fc1 = nn.Linear(153600, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def flatten(self, x):
        # Flatten the input tensor
        N = x.shape[0]
        return x.view(N, -1)

    def forward(self, image, text, device):
        N, C, H, W = image.shape

        # Get visual embeddings using the Region Proposal Network (RPN)
        image_embeds = torch.stack(self.RPN.get_embeds(image))
        visual_token_type_ids = torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.float).to(device)

        # Tokenize and pad the text
        inputs = self.tokenizer(text, padding='max_length', max_length=100, return_tensors="pt").to(device)
        inputs.update(
            {
                "visual_embeds": image_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )

        # Pass the inputs through the VisualBert model
        outputs = self.vbert_model(**inputs)

        # Extract the last hidden state from VisualBert
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.flatten(last_hidden_state)

        # Pass the hidden state through linear layers with normalization and dropout
        fc1_out = self.fc1(last_hidden_state)
        fc1_out = self.ln1(fc1_out)
        fc1_out = self.relu1(fc1_out)
        fc1_out = self.dropout1(fc1_out)

        fc2_out = self.fc2(fc1_out)
        fc2_out = self.ln2(fc2_out)
        fc2_out = self.relu2(fc2_out)
        fc2_out = self.dropout2(fc2_out)

        fc3_out = self.fc3(fc2_out)
        return fc3_out

