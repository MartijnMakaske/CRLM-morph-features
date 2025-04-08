# Imports
import torch
import torch.nn as nn

import glob
import os
import pandas as pd
import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)
import nibabel as nib
#import models
from models import PairedMedicalDataset_Full, PairedMedicalDataset_Images, SiameseNetwork_Full, SiameseNetwork_Images
import matplotlib.pyplot as plt


# Instantiate the base model (e.g., ResNet or other feature extractor)

resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])

model = SiameseNetwork_Images(encoder)
model.load_state_dict(torch.load("images_best_model.pth"))
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

#load images
image1_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR001_0_tumor.nii.gz"
image2_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR001_1_tumor.nii.gz"

# Load the images using nibabel
image1 = nib.load(image1_path).get_fdata()
image2 = nib.load(image2_path).get_fdata()

# Apply the transformations
transform=[ScaleIntensity(), Resize((32, 128, 128), mode="trilinear")]
transform_compose = Compose(transform)
image1 = transform_compose(image1)
image2 = transform_compose(image2)

# Convert to tensors
image1 = torch.tensor(image1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add channel and batch dimensions
image2 = torch.tensor(image2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add channel and batch dimensions

# Make sure images have gradients enabled
img1 = image1.clone().detach().requires_grad_(True)
img2 = image2.clone().detach().requires_grad_(True)

print(img1.shape, img2.shape)

model.eval()
output = model(img1, img2)
target_class = torch.argmax(output, dim=1).item()
score = output[0, target_class]

# Backward for saliency
model.zero_grad()
score.backward()

# Get gradients
saliency_img1 = img1.grad.abs().squeeze().cpu().numpy()
original_img1 = img1.detach().squeeze().cpu().numpy()

# Normalize both for better visualization
saliency_img1 = (saliency_img1 - saliency_img1.min()) / (saliency_img1.max() - saliency_img1.min())
original_img1 = (original_img1 - original_img1.min()) / (original_img1.max() - original_img1.min())

# Choose a slice
slice_idx = saliency_img1.shape[1] // 2  # middle slice
saliency_slice = saliency_img1[:, slice_idx, :]  # shape [H, W]
image_slice = original_img1[:, slice_idx, :]     # shape [H, W]

# Overlay saliency
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_slice, cmap='gray')
plt.title("Original CT Slice")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_slice, cmap='gray')
plt.imshow(saliency_slice, cmap='hot', alpha=0.5)  # overlay
plt.title("Overlay: CT + Saliency")
plt.axis('off')
plt.tight_layout()
plt.show()
