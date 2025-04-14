# Imports
import torch

from utils import SiameseNetwork_Images, PairedMedicalDataset_Images
import glob
import os
import pandas as pd
import numpy as np
from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)

from monai.networks.nets import resnet
from tqdm import tqdm

"""
# Instantiate the base model (e.g., ResNet or other feature extractor)
resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])
"""

encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)

model = SiameseNetwork_Images(encoder)
model.load_state_dict(torch.load("best_model.pth"))
print("Model loaded successfully.")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/scratch/bmep/mfmakaske/training_scans/"
clinical_data_dir = "/scratch/bmep/mfmakaske/"
nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]


pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels.csv"))
all_labels = torch.tensor(pd_labels.values.tolist()) #Fill in correct path. response, PFS, and OS


# Create training and validation datasets
train_dataset = PairedMedicalDataset_Images(
    image_pairs, all_labels, transform=[ScaleIntensity(), Resize((64, 256, 256), mode="trilinear")]
)

# Create DataLoaders
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

probabilities = []
labels = []

model.to(device)

with torch.no_grad():
    for img_1, img_2, label in tqdm(data_loader):
            img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)
            
            # Forward pass
            outputs = model(img_1, img_2)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            

            probabilities.extend(probs.cpu().numpy())
            labels.extend(label.cpu().numpy())
        

# Save probabilities and labels to a CSV file
# Convert probabilities and labels to NumPy arrays
probabilities = np.array(probabilities)  # Shape: [num_samples, num_classes]
labels = np.array(labels)  # Shape: [num_samples, num_classes]

# Create a DataFrame where each column corresponds to a class probability
prob_columns = [f"Prob_Class_{i}" for i in range(probabilities.shape[1])]
label_columns = [f"Label_Class_{i}" for i in range(labels.shape[1])]

output_df = pd.DataFrame(
    np.hstack([probabilities, labels]),  # Combine probabilities and labels horizontally
    columns=prob_columns + label_columns  # Column names for probabilities and labels
)

# Save to CSV
output_df.to_csv("output_probs_labels.csv", index=False)