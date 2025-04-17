# Imports
import torch

import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)
import nibabel as nib

from training_utils import PairedMedicalDataset_Full, PairedMedicalDataset_Images, SiameseNetwork_Full, SiameseNetwork_Images

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
image1 = image1.clone().detach().unsqueeze(0).to(device)  # Add batch dimensions
image2 = image2.clone().detach().unsqueeze(0).to(device)  # Add batch dimensions

# Make sure images have gradients enabled
img1 = image1.clone().detach().requires_grad_(True)
img2 = image2.clone().detach().requires_grad_(True)


feature_maps = {}

def hook_fn(module, input, output):
    feature_maps[module] = output.detach().cpu()

# Choose layers from the encoder you want to visualize
for name, layer in model.base_model.named_modules():
    if "conv1" in name or "conv2" in name:  # You can also look at 'conv1', 'layer2', etc.
        layer.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(img1, img2)  # Forward pass

import matplotlib.pyplot as plt

for layer, fmap in feature_maps.items():
    fmap = fmap[0]  # Take first item in batch, shape [C, D, H, W] or [C, H, W]

    # For 3D: pick a slice from depth
    if fmap.ndim == 4:
        middle = fmap.shape[3] // 2
        fmap = fmap[:, :, :, middle]  # shape [C, H, W]

    # Show first few channels
    num_channels_to_show = min(6, fmap.shape[0])
    fig, axs = plt.subplots(1, num_channels_to_show, figsize=(15, 5))
    fig.suptitle(f"Layer: {layer}")
    
    for i in range(num_channels_to_show):
        axs[i].imshow(fmap[i], cmap="viridis")
        axs[i].axis("off")
    
    plt.show()







"""
# Perform the backward pass for each class
num_classes = output.shape[1]  # Assuming the output shape is (batch_size, num_classes)

# Compute saliency maps for all classes
num_classes = output.shape[1]  # Assuming the output shape is (batch_size, num_classes)
saliency_maps = []

for class_index in range(num_classes):
    # Clear previous gradients
    model.zero_grad()
    
    # Perform backward pass for the current class
    output[:, class_index].backward(retain_graph=True)  # Use retain_graph=True to keep the computation graph
    
    # Extract the gradients for img1
    saliency_img1 = img1.grad.abs().squeeze().cpu().numpy()  # Absolute value of gradients
    
    # Normalize the gradients for better visualization
    saliency_img1 = (saliency_img1 - saliency_img1.min()) / (saliency_img1.max() - saliency_img1.min())
    
    # Store the saliency map for the current class
    saliency_maps.append(saliency_img1)
    
    # Reset gradients for the next iteration
    img1.grad.zero_()

# Initial indices for slice and class
slice_idx = saliency_maps[0].shape[2] // 2  # Start with the middle slice
class_idx = 0  # Start with the first class

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.3)  # Leave space for two sliders

# Display the initial slices

# Convert image1 to a NumPy array for visualization
image1_np = image1.squeeze().cpu().numpy()  # Remove batch and channel dimensions, move to CPU, and convert to NumPy

# Update the slices
image_slice = image1_np[:, :, slice_idx]
saliency_slice = saliency_maps[class_idx][:, :, slice_idx]


img1_plot = ax[0].imshow(image_slice, cmap='gray')
ax[0].set_title("Original CT Slice")
ax[0].axis('off')

img2_plot = ax[1].imshow(image_slice, cmap='gray')
saliency_overlay = ax[1].imshow(saliency_slice, cmap='hot', alpha=0.5)  # Overlay saliency map
ax[1].set_title(f"Overlay: CT + Saliency (Class {class_idx})")
ax[1].axis('off')

# Add a slider for scrolling through slices
ax_slice_slider = plt.axes([0.2, 0.15, 0.6, 0.03])  # Position of the slice slider
slice_slider = Slider(ax_slice_slider, 'Slice', 0, saliency_maps[0].shape[2] - 1, valinit=slice_idx, valstep=1)

# Add a slider for scrolling through classes
ax_class_slider = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position of the class slider
class_slider = Slider(ax_class_slider, 'Class', 0, num_classes - 1, valinit=class_idx, valstep=1)

# Update function for the sliders
def update(val):
    slice_idx = int(slice_slider.val)  # Get the current slice index
    class_idx = int(class_slider.val)  # Get the current class index
    
    # Update the slices
    image_slice = image1_np[:, :, slice_idx]
    saliency_slice = saliency_maps[class_idx][:, :, slice_idx]
    
    # Update the plots
    img1_plot.set_data(image_slice)
    img2_plot.set_data(image_slice)
    saliency_overlay.set_data(saliency_slice)
    ax[1].set_title(f"Overlay: CT + Saliency (Class {class_idx})")
    fig.canvas.draw_idle()  # Redraw the figure

# Connect the sliders to the update function
slice_slider.on_changed(update)
class_slider.on_changed(update)

plt.show()
"""