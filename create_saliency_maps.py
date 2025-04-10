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
from monai.networks.nets import resnet
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


"""
# Instantiate the base model (e.g., ResNet or other feature extractor)
resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])
"""

encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)

model = SiameseNetwork_Images(encoder)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)


#load images
image1_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR017_0_tumor.nii.gz"
image2_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR017_1_tumor.nii.gz"

# Load the images using nibabel
image1 = nib.load(image1_path).get_fdata()
image2 = nib.load(image2_path).get_fdata()

# Add channel dimension for CNN input (C, H, W, D)
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)

# Apply the transformations FIGURE OUT CORRECT RESIZING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
transform=[ScaleIntensity(), Resize((64, 256, 256), mode="trilinear")]
transform_compose = Compose(transform)
image1 = transform_compose(image1)
image2 = transform_compose(image2)

# Convert to tensors
image1 = image1.clone().detach().unsqueeze(0).to(device)  # Add batch dimensions
image2 = image2.clone().detach().unsqueeze(0).to(device)  # Add batch dimensions

# Make sure images have gradients enabled
img1 = image1.clone().detach().requires_grad_(True)
img2 = image2.clone().detach().requires_grad_(True)


model.eval()  # Ensure the model is in evaluation mode if you are only testing

output = model(img1, img2)

# Perform the backward pass
model.zero_grad()  # Clear previous gradients

# Compute the saliency map for a specific class (e.g., class index c)
class_index = 8  # Last class (5y OS)
output[:, class_index].backward()
    
# Extract the gradients for img1
saliency_img1 = img1.grad.abs().squeeze().cpu().numpy()  # Absolute value of gradients
original_img1 = img1.detach().squeeze().cpu().numpy()  # Original image for comparison

# Normalize the gradients for better visualization
saliency_img1 = (saliency_img1 - saliency_img1.min()) / (saliency_img1.max() - saliency_img1.min())
original_img1 = (original_img1 - original_img1.min()) / (original_img1.max() - original_img1.min())

# Initial slice index
slice_idx = saliency_img1.shape[2] // 2  # Start with the middle slice

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.2)  # Leave space for the slider

# Display the initial slices
image_slice = original_img1[:, :, slice_idx]
saliency_slice = saliency_img1[:, :, slice_idx]

img1_plot = ax[0].imshow(image_slice, cmap='gray')
ax[0].set_title("Original CT Slice")
ax[0].axis('off')

img2_plot = ax[1].imshow(image_slice, cmap='gray')
saliency_overlay = ax[1].imshow(saliency_slice, cmap='hot', alpha=0.5)  # Overlay saliency map
ax[1].set_title("Overlay: CT + Saliency")
ax[1].axis('off')

# Add a slider for scrolling through slices
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
slice_slider = Slider(ax_slider, 'Slice', 0, saliency_img1.shape[2] - 1, valinit=slice_idx, valstep=1)

# Update function for the slider
def update(val):
    slice_idx = int(slice_slider.val)  # Get the current slider value
    image_slice = original_img1[:, :, slice_idx]
    saliency_slice = saliency_img1[:, :, slice_idx]
    
    # Update the plots
    img1_plot.set_data(image_slice)
    img2_plot.set_data(image_slice)
    saliency_overlay.set_data(saliency_slice)
    fig.canvas.draw_idle()  # Redraw the figure

# Connect the slider to the update function
slice_slider.on_changed(update)

plt.show()

