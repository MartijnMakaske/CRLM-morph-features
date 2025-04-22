# Imports
import torch

import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Compose,
    Transpose
)
import nibabel as nib

from training_utils import SiameseNetwork_Images_OS

import matplotlib.pyplot as plt
from monai.networks.nets import resnet
import matplotlib.pyplot as plt



"""
# Instantiate the base model (e.g., ResNet or other feature extractor)
resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])
"""

encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)

model = SiameseNetwork_Images_OS(encoder)
model.load_state_dict(torch.load("./models/model_tumor_images_lr3.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

#load images
image1_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR032_0_tumor.nii.gz"
image2_path = "/scratch/bmep/mfmakaske/training_tumor_scans/CAESAR032_1_tumor.nii.gz"

# Load the images using nibabel
image1 = nib.load(image1_path).get_fdata()
image2 = nib.load(image2_path).get_fdata()

# Add channel dimension for CNN input (C, H, W, D)
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)


# Apply the transformations 
transform=[ScaleIntensityRange(a_min=-100, a_max=200, b_min=0.0, b_max=1.0, clip=True), Resize((256, 256, 64), mode="trilinear"),
                                                Transpose((0, 3, 2, 1))]
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



for layer, fmap in feature_maps.items():
    fmap = fmap[0]  # Take first item in batch, shape [C, D, H, W] or [C, H, W]


    # For 3D: pick a slice from depth
    if fmap.ndim == 4:
        middle = fmap.shape[1] // 2
        fmap = fmap[:, middle, :, :]  # shape [D, H, W]

    # Show first few channels
    num_channels_to_show = min(6, fmap.shape[0])
    fig, axs = plt.subplots(1, num_channels_to_show  + 1 , figsize=(15, 5))
    fig.suptitle(f"Layer: {layer}")

    # Show the original image slice
    original_image_slice = img1[0, 0, 32, :, :].detach().cpu().numpy()  # Extract the same slice as the feature maps
    axs[6].imshow(original_image_slice, cmap="gray")
    axs[6].set_title("Original Image")
    axs[6].axis("off")
    
    for i in range(num_channels_to_show):
        axs[i].imshow(fmap[i], cmap="viridis")
        axs[i].axis("off")
    
    plt.show()
