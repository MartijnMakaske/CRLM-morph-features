# Imports
import torch

import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)
import nibabel as nib

from utils import PairedMedicalDataset_Full, PairedMedicalDataset_Images, SiameseNetwork_Full, SiameseNetwork_Images

import matplotlib.pyplot as plt
from monai.networks.nets import resnet
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
import torch.nn as nn

"""
# Instantiate the base model (e.g., ResNet or other feature extractor)
resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])
"""

class WrappedSiameseModel(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.model = siamese_model

    def forward(self, input_combined):
        # input_combined shape: [1, 2, D, H, W]
        img1 = input_combined[:, 0:1, ...]
        img2 = input_combined[:, 1:2, ...]
        return self.model(img1, img2)


encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)

model = SiameseNetwork_Images(encoder)
model.load_state_dict(torch.load("./models/model_full_images_lr3.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

#load images
image1_path = "/scratch/bmep/mfmakaske/training_scans/CAESAR032_0.nii.gz"
image2_path = "/scratch/bmep/mfmakaske/training_scans/CAESAR032_1.nii.gz"

# Load the images using nibabel
image1 = nib.load(image1_path).get_fdata()
image2 = nib.load(image2_path).get_fdata()

# Add channel dimension for CNN input (C, H, W, D)  (perhaps add later?)
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)

# Apply the transformations 
transform=[ScaleIntensity(), Resize((256, 256, 64), mode="trilinear")]
transform_compose = Compose(transform)
image1 = transform_compose(image1)
image2 = transform_compose(image2)

# Convert to tensors
image1 = image1.clone().detach().unsqueeze(0)  # Add batch dimensions
image2 = image2.clone().detach().unsqueeze(0)  # Add batch dimensions

# Make sure images have gradients enabled
img1 = image1.clone().detach().requires_grad_(True)
img2 = image2.clone().detach().requires_grad_(True)


model.eval()  # Ensure the model is in evaluation mode if you are only testing

# Inputs
input_combined = torch.cat([img1, img2], dim=1).to(device) # shape: [1, 2, D, H, W]
input_combined.requires_grad_()

# Wrap the model
wrapped_model = WrappedSiameseModel(model).to(device)
wrapped_model.eval()

ig = IntegratedGradients(wrapped_model)
num_samples = 2
stdev = 0.1
target_class = 3  # pick your label

# Allocate space
attr_sum = torch.zeros_like(input_combined).to(device)

for i in range(num_samples):
    noise = torch.normal(0, stdev, size=input_combined.shape).to(device)
    noisy_input = (input_combined + noise).requires_grad_()
    
    attr = ig.attribute(noisy_input, target=target_class)
    attr_sum += attr

attributions = attr_sum / num_samples
