import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipClassifier(nn.Module):
    def __init__(self, settings, clip_model=None):
        super(ClipClassifier, self).__init__()
        
        # Initialize the CLIP model (provided as 'clip_model')
        self.clip = clip_model
        
        # Dropout probability
        self.pdrop = settings['pdrop']

        # Define a linear classifier layer with 1024 input features and 1 output feature (binary classification)
        self.classifier = nn.Linear(1024, 1).half()  # 'half()' indicates the use of half-precision floating-point arithmetic
        
    def forward(self, qimage_clip_processed, qtext_clip_tokenized):
        # Encode the image and text using the CLIP model
        encoded_img = self.clip.encode_image(qimage_clip_processed)
        encoded_text = self.clip.encode_text(qtext_clip_tokenized)

        # Normalize the encoded features
        encoded_img = encoded_img / encoded_img.norm(dim=-1, keepdim=True)
        encoded_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

        # Concatenate the normalized image and text features
        joint_features = torch.cat((encoded_img, encoded_text), dim=1)

        # Apply dropout with the specified dropout probability
        joint_features = F.dropout(joint_features, p=self.pdrop)

        # Pass the joint features through the linear classifier
        consis_out = self.classifier(joint_features)

        return consis_out
