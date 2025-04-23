#imports
import os
import pandas as pd
import glob

from tqdm import tqdm
from monai.networks.nets import resnet
from training_utils import PairedMedicalDataset_Images, SiameseNetwork_Images_OS

from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Transpose,
    EnsureType
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
import wandb

from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

# Optimize for performance with torch.compile
torch.set_float32_matmul_precision('high')


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):

    global run

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
        all_labels_train = []
        all_probs_train = []

        for train_img_1, train_img_2, train_labels in tqdm(train_loader):
            train_img_1, train_img_2, train_labels = train_img_1.to(device), train_img_2.to(device), train_labels.to(device)

            # Forward pass
            outputs = model(train_img_1, train_img_2)

            train_labels = train_labels.squeeze(1).long()
            loss = criterion(outputs, train_labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save outputs for AUC-ROC
            probs = F.softmax(outputs, dim=1).detach().cpu()
            labels = train_labels.detach().cpu()

            all_probs_train.append(probs)
            all_labels_train.append(labels)

            running_loss += loss     
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == train_labels).sum().item()
            total += train_labels.size(0)

            # Free memory explicitly
            del train_img_1, train_img_2, train_labels, outputs, preds
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        # Stack all predictions and labels
        all_probs_train = torch.cat(all_probs_train).numpy()
        all_labels_train = torch.cat(all_labels_train).numpy()

        # One-hot encode the labels for multiclass AUC
        all_labels_train_oh = label_binarize(all_labels_train, classes=[0,1,2,3])

        # Compute AUC-ROC
        try:
            epoch_auc = roc_auc_score(all_labels_train_oh, all_probs_train, average="macro", multi_class="ovr")
        except ValueError:
            epoch_auc = float('nan')  # In case only 1 class is present in the batch

        # Log metrics to wandb
        run.log({"train acc": epoch_acc, "train loss": epoch_loss, "train auc": epoch_auc})

        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Train AUC: {epoch_auc:.4f}')
        validate_model(model, val_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step()


def validate_model(model, val_loader, criterion, device="cuda"):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    global best_val_acc
    global model_name
    global run

    all_preds = []
    all_ground_truths = []
    all_probs_val = []

    with torch.no_grad():
        for val_img_1, val_img_2, val_labels in val_loader:
            val_img_1, val_img_2, val_labels = val_img_1.to(device), val_img_2.to(device), val_labels.to(device)

            # Forward pass
            outputs = model(val_img_1, val_img_2)

            val_labels = val_labels.squeeze(1).long()
            loss = criterion(outputs, val_labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            # Save outputs for AUC-ROC
            probs = F.softmax(outputs, dim=1).detach().cpu()

            all_probs_val.append(probs)

            correct += (preds == val_labels).sum().item()
            total += val_labels.size(0)

            # Collect predictions and labels for F1 score
            all_preds.extend(preds.cpu().numpy())
            all_ground_truths.extend(val_labels.cpu().numpy())

            # Free memory explicitly
            del val_img_1, val_img_2, val_labels, outputs, preds
            torch.cuda.empty_cache()
            
    # Accuracy
    val_loss /= len(val_loader)
    val_acc = correct / total

    f1 = f1_score(all_ground_truths, all_preds, average="weighted")  # Use "weighted" for class imbalance

    # Stack all predictions and labels
    all_probs_val = torch.cat(all_probs_val).numpy()

    # One-hot encode the labels for multiclass AUC
    all_ground_truths_val_oh = label_binarize(all_ground_truths, classes=[0,1,2,3])

    # Compute AUC-ROC
    try:
        val_auc = roc_auc_score(all_ground_truths_val_oh, all_probs_val, average="macro", multi_class="ovr")
    except ValueError:
        val_auc = float('nan')  # In case only 1 class is present in the batch


    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Validation F1 score: {f1:.4f} | Validation AUC: {val_auc:.4f}")

    # Log metrics to wandb
    run.log({"val acc": val_acc, "val loss": val_loss, "val f1": f1, "val auc": val_auc})

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model._orig_mod.state_dict(), f'./models/{model_name}.pth')
        print("Best model saved!")
   


# ---------------------------------------
# READ DATA
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/scratch/bmep/mfmakaske/training_tumor_scans/"
clinical_data_dir = "/scratch/bmep/mfmakaske/"

#data_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/paired_tumor_scans"
#clinical_data_dir = "C:/Users/P095550/Documents/CRLM-morph-features/CRLM-morph-features"

nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]

pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels_OS.csv"))
all_labels = torch.tensor(pd_labels.values.tolist())

# ---------------------------------------
# HYPERPARAMETERS
# ---------------------------------------
batch_size = 8
num_epochs = 100
learning_rate = 1e-3
model_name = "model_full_images_lr3_100epochs"

class_weights = torch.tensor([1.6363636363636365, 0.496551724137931, 0.96, 3.0], dtype=torch.float32).to(device)

#-------------------------------------
# TRACK WITH WANDB
#-------------------------------------

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="martijnmakaske-vrije-universiteit-amsterdam",
    # Set the wandb project where this run will be logged.
    project="CRLM-morph-features",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "architecture": "Siames-Morph-CNN",
        "dataset": "CAIRO scaled images",
        "epochs": num_epochs,
    },
)


# Initialize best validation accuracy
best_val_acc = 0.0

# Perform a single train-test split with stratification
train_image_pairs, val_image_pairs, train_labels, val_labels = train_test_split(
    image_pairs, all_labels, test_size=0.2, random_state=42
)

# Create training and validation datasets
train_dataset = PairedMedicalDataset_Images(
    train_image_pairs, train_labels, transform=[ScaleIntensityRange(a_min=-100,
                                                                     a_max=200, b_min=0.0, b_max=1.0, clip=True), 
                                                Resize((128, 128, 64), 
                                                mode="trilinear"),
                                                Transpose((0, 3, 2, 1)),
                                                EnsureType(data_type="tensor")]
)
val_dataset = PairedMedicalDataset_Images(
    val_image_pairs, val_labels, transform=[ScaleIntensityRange(a_min=-100,
                                                                 a_max=200, b_min=0.0, b_max=1.0, clip=True), 
                                            Resize((128, 128, 64), 
                                            mode="trilinear"),
                                            Transpose((0, 3, 2, 1)),
                                            EnsureType(data_type="tensor")]
)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize the model
encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)

# Freeze the weights for the first 3 layers of the encoder
"""
for name, layer in encoder.named_children():
    if "layer4" not in name:  # Freeze the first 3 layers
        for param in layer.parameters():
            param.requires_grad = False
        print(f"Froze layer: {name}")


# print which layers are trainable and which are frozen
for name, param in encoder.named_parameters():
    print(f"Layer: {name} | requires_grad: {param.requires_grad}")
"""

model = torch.compile(SiameseNetwork_Images_OS(encoder))
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer (when overfitting, use weight decay or AdamW)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size= (num_epochs//2) , gamma=0.1)

# Train the model for this fold
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

run.finish()
print("\nTraining and Validation Complete.")