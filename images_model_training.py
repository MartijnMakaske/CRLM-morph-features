#imports
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from monai.networks.nets import resnet
from utils import PairedMedicalDataset_Images, SiameseNetwork_Images
from torchmetrics.classification import MultilabelHammingDistance

from monai.transforms import (
    Resize,
    ScaleIntensity,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score



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
        
        for train_img_1, train_img_2, train_labels in tqdm(train_loader):
            train_img_1, train_img_2, train_labels = train_img_1.to(device), train_img_2.to(device), train_labels.to(device)
            
            # Forward pass
            outputs = model(train_img_1, train_img_2)

            # Create a mask where labels are not -1
            loss_mask = (train_labels != -1).float()  # 1 where label != -1, 0 where label == -1

            # Apply the mask to the loss
            loss = criterion(outputs, train_labels) * loss_mask
            loss = loss.sum() / loss_mask.sum()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Apply threshold to get binary predictions
            preds = (probs > 0.5).float()
            
            # Masked accuracy
            correct = ((preds == train_labels) * loss_mask).sum()
            total = loss_mask.sum()

            # Free memory
            del train_img_1, train_img_2, train_labels, outputs, probs, preds, loss_mask
            torch.cuda.empty_cache()


        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f'Training complete. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        validate_model(model, val_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step()


def validate_model(model, val_loader, criterion, device="cuda"):
    model.eval()
    global best_val_loss
    val_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_ground_truths = []

    metric = MultilabelHammingDistance(num_labels=9).to(device)

    with torch.no_grad():
        for val_img_1, val_img_2, val_labels in val_loader:
            val_img_1, val_img_2, val_labels = val_img_1.to(device), val_img_2.to(device), val_labels.to(device)
            
            # Forward pass
            outputs = model(val_img_1, val_img_2)

            # Create a mask where labels are not -1
            loss_mask = (val_labels != -1).float()  # 1 where label != -1, 0 where label == -1

            # Apply the mask to the loss
            loss = criterion(outputs, val_labels) * loss_mask
            loss = loss.sum() / loss_mask.sum()

            val_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Store predictions and labels for AUC-ROC calculation
            all_probs.append(probs.cpu().numpy())
            all_ground_truths.append(val_labels.cpu().numpy())

            # Apply threshold to get binary predictions
            preds = (probs > 0.5).float()

            # Masked accuracy
            correct = ((preds == val_labels) * loss_mask).sum()
            total = loss_mask.sum()

            metric.update((preds * loss_mask).int(), (val_labels * loss_mask).int())

            # Free memory
            del val_img_1, val_img_2, val_labels, outputs, probs, preds, loss_mask
            torch.cuda.empty_cache()
            
    # Accuracy
    val_loss /= len(val_loader)
    val_acc = correct / total

    # Concatenate all predictions and labels
    all_probs = np.vstack(all_probs)
    all_ground_truths = np.vstack(all_ground_truths)

    # Mask out invalid labels
    masked_probs = all_probs[all_ground_truths != -1]
    masked_ground_truths = all_ground_truths[all_ground_truths != -1]

    # Hamming loss
    hloss = metric.compute()
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(masked_ground_truths, masked_probs, average="macro")
    except ValueError:
        auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Validation Hamming loss: {hloss:.4f} | AUC-ROC: {auc_roc:.4f}")
    #metrics_per_outcome(masked_ground_truths, masked_probs)

    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")


def metrics_per_outcome(masked_ground_truths, masked_probs):
    """
    Currently does not work
    """
    outcomes = {"path_resp": 0, "PFS": 3, "OS": 6}
    for (outcome, i) in outcomes.items():
        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(masked_ground_truths[:,i:i+3], masked_probs[:, i:i+3], average="macro")
        except ValueError:
            auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated
        print(f"{outcome} - AUC-ROC: {auc_roc}")


# ---------------------------------------
# READ DATA
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/scratch/bmep/mfmakaske/training_tumor_scans/"
clinical_data_dir = "/scratch/bmep/mfmakaske/"
nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]

pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels.csv"))
all_labels = torch.tensor(pd_labels.values.tolist()) #Fill in correct path. response, PFS, and OS


#------------------------------------
# INITIALIZE ENCODER
#------------------------------------

#resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet50')

# Remove the final classification layer (fc) to keep only the encoder part
#encoder = nn.Sequential(*list(resnet_model.children())[:-1])

# HYPERPARAMETERS
n_splits = 5
batch_size = 2
num_epochs = 30
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
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]

    # Create training and validation datasets
    train_dataset = PairedMedicalDataset_Images(
        train_image_pairs, train_labels, transform=[ScaleIntensity(), Resize((64, 256, 256), mode="trilinear")]
    )
    val_dataset = PairedMedicalDataset_Images(
        val_image_pairs, val_labels, transform=[ScaleIntensity(), Resize((64, 256, 256), mode="trilinear")]
    )


    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize the model for this fold
    model = SiameseNetwork_Images(resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True))
    model = model.to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model for this fold
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

# Calculate average metrics across all folds
avg_val_loss = sum([result["val_loss"] for result in fold_results]) / n_splits
avg_val_acc = sum([result["val_acc"] for result in fold_results]) / n_splits

print("\nCross-Validation Results:")
print(f"Average Val Loss: {avg_val_loss:.4f}")
print(f"Average Val Acc: {avg_val_acc:.4f}")