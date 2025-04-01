#imports
import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)

import glob
import nibabel as nib
from sklearn.model_selection import train_test_split
from monai.data import Dataset

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import copy
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#----------------------------------
# CLASSES
#----------------------------------

class PairedMedicalDataset(Dataset):
    def __init__(self, image_pairs, metadata, labels, transform=None):
        self.image_pairs = image_pairs
        self.metadata = metadata
        self.labels = labels
        self.transform = Compose(transform)
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        
        # Load images using nibabel (for NIfTI)
        img1 = nib.load(img1_path).get_fdata()
        img2 = nib.load(img2_path).get_fdata()

        # Add channel dimension for CNN input (C, H, W, D)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        
        metadata = self.metadata[idx]
        label = self.labels[idx]

        label = label.float()
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, metadata, label
    
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2054,256),
            nn.ReLU(),
            nn.Linear(256,9)    # No softmax, because BCEWithLogitsLoss applies sigmoid internally
        )

    
    def forward(self, image1, image2, metadata):
        # Pass both inputs through the shared model
        output1 = self.base_model(image1)
        output2 = self.base_model(image2)

        # Apply adaptive average pooling to both outputs
        output1 = self.adaptive_pool(output1).view(output1.size(0), -1)  # Flatten after pooling
        output2 = self.adaptive_pool(output2).view(output2.size(0), -1)  # Flatten after pooling

        combined_embeddings = torch.cat((output1, output2, metadata), dim=1)
        output3 = self.classifier(combined_embeddings)
        return output3



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/paired_scans"
clinical_data_dir = "C:/Users/P095550/OneDrive - Amsterdam UMC/Documenten/GitHub/CRLM-morph-features"
nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]

pd_metadata = pd.read_csv(os.path.join(clinical_data_dir, "training_input.csv"))
metadata = torch.tensor(pd_metadata.values.tolist())

pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels.csv"))
labels = torch.tensor(pd_labels.values.tolist()) #Fill in correct path. response, PFS, and OS


#------------------------------------
# INITIALIZE ENCODER
#------------------------------------

resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet50')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])



def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 20)

        # ---------------------------
        # TRAINING PHASE
        # ---------------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for img_1, img_2, metadata, labels in tqdm(train_loader):
            img_1, img_2, metadata, labels = img_1.to(device), img_2.to(device), metadata.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(img_1, img_2, metadata)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Apply threshold to get binary predictions
            preds = (probs > 0.5).float()
            
            # Calculate metrics
            correct += (preds == labels).sum().item()
            total += labels.numel()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f'Training complete. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')

        # ---------------------------
        # LEARNING RATE SCHEDULER STEP
        # ---------------------------
        scheduler.step()


# Number of folds
n_splits = 5

batch_size = 1
num_epochs = 1

# Initialize KFold
mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
fold_results = []
best_val_loss = float('inf')

print(f"length image pairs: {len(image_pairs)}")
print(f"length metadata: {len(metadata)}")
print(f"Length labels: {len(labels)}")

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(mlskf.split(image_pairs, labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    print("-" * 20)

    print(train_idx)
    print(val_idx)


    # Split the data into training and validation sets for this fold
    train_image_pairs = [image_pairs[i] for i in train_idx]
    val_image_pairs = [image_pairs[i] for i in val_idx]
    train_metadata = metadata[train_idx]
    val_metadata = metadata[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    # Create training and validation datasets
    train_dataset = PairedMedicalDataset(
        train_image_pairs, train_metadata, train_labels, transform=[ScaleIntensity(), Resize((64, 256, 256))]
    )
    val_dataset = PairedMedicalDataset(
        val_image_pairs, val_metadata, val_labels, transform=[ScaleIntensity(), Resize((64, 256, 256))]
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model for this fold
    model = SiameseNetwork(copy.deepcopy(encoder))
    model = model.to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model for this fold
    #train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)
    """
    # Evaluate the model on the validation set
    val_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for img_1, img_2, metadata, labels in val_loader:
            img_1, img_2, metadata, labels = img_1.to(device), img_2.to(device), metadata.to(device), labels.to(device)
            outputs = model(img_1, img_2, metadata)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Apply sigmoid and threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Calculate metrics
            correct += (preds == labels).sum().item()
            total += labels.numel()

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Fold {fold + 1} - Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Store results for this fold
    fold_results.append({"fold": fold + 1, "val_loss": val_loss, "val_acc": val_acc})

    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")
    """
# Calculate average metrics across all folds
avg_val_loss = sum([result["val_loss"] for result in fold_results]) / n_splits
avg_val_acc = sum([result["val_acc"] for result in fold_results]) / n_splits

print("\nCross-Validation Results:")
print(f"Average Val Loss: {avg_val_loss:.4f}")
print(f"Average Val Acc: {avg_val_acc:.4f}")