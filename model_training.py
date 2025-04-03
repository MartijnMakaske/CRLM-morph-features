#imports
import os
import numpy as np
import pandas as pd
import glob
import nibabel as nib
import copy
from tqdm import tqdm

from monai.data import Dataset
from monai.transforms import (
    Resize,
    ScaleIntensity,
    Compose
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score

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
            nn.Linear(518,256),     #use 2054 for resnet50
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

# ---------------------------------------
# READ DATA
# ---------------------------------------

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


#------------------------------------
# INITIALIZE ENCODER
#------------------------------------

resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet10')

# Remove the final classification layer (fc) to keep only the encoder part
encoder = nn.Sequential(*list(resnet_model.children())[:-1])



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):

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
        
        for train_img_1, train_img_2, train_metadata, train_labels in tqdm(train_loader):
            train_img_1, train_img_2, train_metadata, train_labels = train_img_1.to(device), train_img_2.to(device), train_metadata.to(device), train_labels.to(device)
            
            # Forward pass
            outputs = model(train_img_1, train_img_2, train_metadata)
            loss = criterion(outputs, train_labels)
            
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
            correct += (preds == train_labels).sum().item()
            total += train_labels.numel()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f'Training complete. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        validate_model(model, val_loader, criterion, device)

        # ---------------------------
        # LEARNING RATE SCHEDULER STEP
        # ---------------------------
        scheduler.step()


def validate_model(model, val_loader, criterion, device="cuda"):
    model.eval()
    global best_val_loss
    val_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_ground_truths = []

    with torch.no_grad():
        for val_img_1, val_img_2, val_metadata, val_labels in val_loader:
            val_img_1, val_img_2, val_metadata, val_labels = val_img_1.to(device), val_img_2.to(device), val_metadata.to(device), val_labels.to(device)
            
            # Forward pass
            outputs = model(val_img_1, val_img_2, val_metadata)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Store predictions and labels for AUC-ROC calculation
            all_probs.append(probs.cpu().numpy())
            all_ground_truths.append(val_labels.cpu().numpy())

            # Apply threshold to get binary predictions
            preds = (probs > 0.5).float()

            # Calculate metrics
            correct += (preds == val_labels).sum().item()
            total += val_labels.numel()

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Concatenate all predictions and labels
    all_probs = np.vstack(all_probs)
    all_ground_truths = np.vstack(all_ground_truths)

    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(all_ground_truths, all_probs, average="macro")
    except ValueError:
        auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | AUC-ROC: {auc_roc:.4f}")
    metrics_per_outcome(all_ground_truths, all_probs)

    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")


def metrics_per_outcome(all_ground_truths, all_probs):
    outcomes = {"path_resp": 0, "PFS": 3, "OS": 6}
    for (outcome, i) in outcomes.items():
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(all_ground_truths[:, i:i+3], all_probs[:, i:i+3], average="macro")
        except ValueError:
            auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated
        print(f"{outcome} - AUC-ROC: {auc_roc}")









# HYPERPARAMETERS
n_splits = 5
batch_size = 4
num_epochs = 50
class_weights = torch.tensor([ 5.7600,  8.0000, 10.2857,  1.2152,  2.4202,  9.0000,  1.1803,  2.9091,
                            12.0000], dtype=torch.float32).to(device)

# Initialize KFold
mlskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
fold_results = []
best_val_loss = float('inf')

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(mlskf.split(image_pairs, all_labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    print("-" * 20)

    # Split the data into training and validation sets for this fold
    train_image_pairs = [image_pairs[i] for i in train_idx]
    val_image_pairs = [image_pairs[i] for i in val_idx]
    train_metadata = all_metadata[train_idx]
    val_metadata = all_metadata[val_idx]
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

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
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model for this fold
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)
    """
    # Evaluate the model on the validation set
    val_loss = 0.0
    correct = 0
    total = 0

    # Initialize lists to store predictions and ground truth
    all_probs = []
    all_ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for val_img_1, val_img_2, val_metadata, val_labels in val_loader:
            val_img_1, val_img_2, val_metadata, val_labels = val_img_1.to(device), val_img_2.to(device), val_metadata.to(device), val_labels.to(device)
            outputs = model(val_img_1, val_img_2, val_metadata)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()

            # Apply sigmoid
            probs = torch.sigmoid(outputs)

            # Store predictions and labels for AUC-ROC calculation
            all_probs.append(probs.cpu().numpy())
            all_ground_truths.append(val_labels.cpu().numpy())

            #apply threshold
            preds = (probs > 0.5).float()

            # Calculate metrics
            correct += (preds == val_labels).sum().item()
            total += val_labels.numel()

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Concatenate all predictions and labels
    all_probs = np.vstack(all_probs)
    all_ground_truths = np.vstack(all_ground_truths)

    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(all_ground_truths, all_probs, average="macro")
    except ValueError:
        auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated
    
    # Store results for this fold
    print(f"Fold {fold + 1} ---------------------------")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | AUC-ROC: {auc_roc:.4f}")

    metrics_per_outcome(all_ground_truths, all_probs)

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