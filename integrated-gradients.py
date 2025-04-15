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
import torch.nn as nn
import matplotlib.pyplot as plt

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


# Assume `img1` and `img2` are both shaped [1, 1, D, H, W]
input_combined = torch.cat([img1, img2], dim=1).requires_grad_()  # shape: [1, 2, D, H, W]


wrapped_model = WrappedSiameseModel(model).to(device).eval()
input_combined = input_combined.to(device)

# Choose target class (e.g., class index 3)
target_class = 6

# Run IG
ig = IntegratedGradients(wrapped_model)

baseline = torch.zeros_like(input_combined).to(device)
attributions = ig.attribute(input_combined, baselines=baseline, target=target_class, n_steps=20, internal_batch_size=2)


# For the first input image
img1_attr = attributions[0, 0].detach().cpu()  # [D, H, W]

# For the second input image
img2_attr = attributions[0, 1].detach().cpu()

from matplotlib.widgets import Slider

# Normalize the saliency map for visualization
img1_attr_normalized = (img1_attr - img1_attr.min()) / (img1_attr.max() - img1_attr.min())
img2_attr_normalized = (img2_attr - img2_attr.min()) / (img2_attr.max() - img2_attr.min())

# Convert the original images to NumPy arrays for visualization
image1_np = img1.squeeze().detach().cpu().numpy()  # Remove batch and channel dimensions
image2_np = img2.squeeze().detach().cpu().numpy()

# Initial slice index
slice_idx = img1_attr.shape[2] // 2  # Start with the middle slice

# Create the figure and axes
fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns
plt.subplots_adjust(bottom=0.2)  # Leave space for the slider

# Display the initial slices
image1_slice = image1_np[:, :, slice_idx]
saliency1_slice = img1_attr_normalized[:, :, slice_idx]

image2_slice = image2_np[:, :, slice_idx]
saliency2_slice = img2_attr_normalized[:, :, slice_idx]

# Plot the first original image
original_img1_plot = ax[0, 0].imshow(image1_slice, cmap='gray')
ax[0, 0].set_title(f"Original Image 1 (Slice {slice_idx})")
ax[0, 0].axis('off')

# Plot the second original image
original_img2_plot = ax[0, 1].imshow(image2_slice, cmap='gray')
ax[0, 1].set_title(f"Original Image 2 (Slice {slice_idx})")
ax[0, 1].axis('off')

# Plot the first image with saliency overlay
img1_plot = ax[1, 0].imshow(image1_slice, cmap='gray')
saliency1_overlay = ax[1, 0].imshow(saliency1_slice, cmap='hot', alpha=0.5)  # Overlay saliency map
ax[1, 0].set_title(f"Image 1 + Saliency (Slice {slice_idx})")
ax[1, 0].axis('off')

# Plot the second image with saliency overlay
img2_plot = ax[1, 1].imshow(image2_slice, cmap='gray')
saliency2_overlay = ax[1, 1].imshow(saliency2_slice, cmap='hot', alpha=0.5)  # Overlay saliency map
ax[1, 1].set_title(f"Image 2 + Saliency (Slice {slice_idx})")
ax[1, 1].axis('off')

# Add a slider for scrolling through slices
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the slider
slice_slider = Slider(ax_slider, 'Slice', 0, img1_attr.shape[2] - 1, valinit=slice_idx, valstep=1)

# Update function for the slider
def update(val):
    slice_idx = int(slice_slider.val)  # Get the current slice index

    # Update the slices
    image1_slice = image1_np[:, :, slice_idx]
    saliency1_slice = img1_attr_normalized[:, :, slice_idx]

    image2_slice = image2_np[:, :, slice_idx]
    saliency2_slice = img2_attr_normalized[:, :, slice_idx]

    # Update the original image plots
    original_img1_plot.set_data(image1_slice)
    original_img2_plot.set_data(image2_slice)

    # Update the saliency overlay plots
    img1_plot.set_data(image1_slice)
    saliency1_overlay.set_data(saliency1_slice)
    ax[1, 0].set_title(f"Image 1 + Saliency (Slice {slice_idx})")

    img2_plot.set_data(image2_slice)
    saliency2_overlay.set_data(saliency2_slice)
    ax[1, 1].set_title(f"Image 2 + Saliency (Slice {slice_idx})")

    fig.canvas.draw_idle()  # Redraw the figure

# Connect the slider to the update function
slice_slider.on_changed(update)

plt.show()