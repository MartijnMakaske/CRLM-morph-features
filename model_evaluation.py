# Imports
import torch
import torch.nn as nn
from models import SiameseNetwork, PairedMedicalDataset
import shap
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


# Instantiate the base model (e.g., ResNet or other feature extractor)

resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])

model = SiameseNetwork(encoder)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

class SiameseWrapper(nn.Module):
    def __init__(self, model):
        super(SiameseWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        image1, image2, metadata = input
        return self.model(image1, image2, metadata)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/scratch/bmep/mfmakaske/training_scans/"
clinical_data_dir = "/scratch/bmep/mfmakaske/"
nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]

pd_metadata = pd.read_csv(os.path.join(clinical_data_dir, "training_input.csv"))
all_metadata = torch.tensor(pd_metadata.values.tolist())

pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels.csv"))
all_labels = torch.tensor(pd_labels.values.tolist()) #Fill in correct path. response, PFS, and OS



# Create training and validation datasets
train_dataset = PairedMedicalDataset(
    image_pairs, all_metadata, all_labels, transform=[ScaleIntensity(), Resize((64, 256, 256), mode="trilinear")]
)

# Create DataLoaders
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=11, shuffle=True)

#data

# Create an iterator for the DataLoader
data_iter = iter(data_loader)

# Extract one batch
batch = next(data_iter)

# Unpack the batch (if applicable)
img1, img2, metadata, labels = batch


# Dummy image pair input
image1_test = img1[0].to("cuda")
image2_test = img2[0].to("cuda")
metadata_test = metadata[0].to("cuda")


# Background for SHAP (used by DeepExplainer)
image1_bg = img1[1:].to("cuda")
image2_bg = img2[1:].to("cuda")
metadata_bg = metadata[1:].to("cuda")

wrapped_model = SiameseWrapper(model).to("cuda")

# Use DeepExplainer for PyTorch models
explainer = shap.GradientExplainer(
    wrapped_model,
    data=(image1_bg, image2_bg, metadata_bg)
)

shap_values = explainer.shap_values((image1_test, image2_test, metadata_test))