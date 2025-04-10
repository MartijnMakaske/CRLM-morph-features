#imports
import os
import numpy as np
import pandas as pd
import glob
import copy
from tqdm import tqdm
from monai.networks.nets import resnet
from models import PairedMedicalDataset_Images, SiameseNetwork_Images
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
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np





def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):

    # After the backward pass, check the gradients of the encoder's parameters
    def check_gradients(model):
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only check the parameters that require gradients
                if param.grad is None:
                    print(f"WARNING: {name} has no gradient!")
                elif param.grad.abs().sum() == 0:
                    print(f"WARNING: {name} has zero gradient!")
                else:
                    print(f"Gradients for {name}: {param.grad.abs().sum()}")


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
        # After the backward pass, check the gradients of the encoder's parameters

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

            check_gradients(model)  # Check gradients after backward pass




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

    per_class_auc, mean_auc = compute_masked_multilabel_auc_roc(all_probs, all_ground_truths)
    print(f"Per-class AUC-ROC: {per_class_auc}")

    
    """
    # Mask out invalid labels
    masked_probs = all_probs[all_ground_truths != -1]
    masked_ground_truths = all_ground_truths[all_ground_truths != -1]
    """
    # Hamming loss
    hloss = metric.compute()
    """
    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(masked_ground_truths, masked_probs, average="macro")
    except ValueError:
        auc_roc = float('nan')  # Handle cases where AUC-ROC cannot be calculated
    """
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Validation Hamming loss: {hloss:.4f} | AUC-ROC: {mean_auc:.4f}")
    #metrics_per_outcome(masked_ground_truths, masked_probs)
    #show_roc_curve(masked_ground_truths, masked_probs)


    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved!")

import numpy as np
from sklearn.metrics import roc_auc_score

def compute_masked_multilabel_auc_roc(all_probs, all_ground_truths, mask_value=-1):
    """
    Computes per-class and average AUC-ROC for multi-label classification,
    handling missing labels using a mask value (e.g., -1).

    Args:
        all_probs (np.ndarray): Predicted probabilities, shape [N, C]
        all_ground_truths (np.ndarray): Ground truth labels, shape [N, C]
        mask_value (int or float): Value in labels to be ignored (e.g., -1 for missing)

    Returns:
        per_class_auc (list): AUC-ROC for each class (NaN if not computable)
        mean_auc (float): Mean AUC-ROC across valid classes
    """
    assert all_probs.shape == all_ground_truths.shape, "Shape mismatch between predictions and labels."

    valid_mask = all_ground_truths != mask_value
    num_classes = all_probs.shape[1]

    print(num_classes)

    per_class_auc = []

    for i in range(num_classes):
        y_true = all_ground_truths[:, i][valid_mask[:, i]]
        y_score = all_probs[:, i][valid_mask[:, i]]
    
        if len(np.unique(y_true)) > 1:
            print(f"Computing AUC for class {i} with {len(y_true)} samples.")
            auc = roc_auc_score(y_true, y_score)
        else:
            print(f"Not enough data to compute AUC for class {i}.")
            auc = np.nan  # Not enough data to compute AUC
        per_class_auc.append(auc)

    mean_auc = np.nanmean(per_class_auc)

    return per_class_auc, mean_auc


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

def show_roc_curve(masked_ground_truths, masked_probs):
    # Create ROC curve for each label
    for i in range(masked_ground_truths.shape[1]):
        fpr, tpr, _ = roc_curve(masked_ground_truths[:, i], masked_probs[:, i])
        plt.plot(fpr, tpr, label=f'Label {i} (AUC = {auc(fpr, tpr):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('roc_curve.png')
    plt.close()

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
#n_splits = 5
batch_size = 2
num_epochs = 100
class_weights = torch.tensor([ 5.7600,  8.0000, 10.2857,  1.2152,  2.4202,  9.0000,  1.1803,  2.9091,
                            12.0000], dtype=torch.float32).to(device)


# Store metrics for each fold
best_val_loss = float('inf')

# Perform a single train-test split with stratification
train_image_pairs, val_image_pairs, train_labels, val_labels = train_test_split(
    image_pairs, all_labels, test_size=0.2, random_state=42
)

print("Training label distribution:", train_labels.sum(dim=0))
print("Validation label distribution:", val_labels.sum(dim=0))

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

# Initialize the model
model = SiameseNetwork_Images(resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True))
model = model.to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# Train the model for this fold
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

print("\nTraining and Validation Complete.")