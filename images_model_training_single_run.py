#imports
import os
import numpy as np
import pandas as pd
import glob

from tqdm import tqdm
from monai.networks.nets import resnet
from training_utils import PairedMedicalDataset_Images, SiameseNetwork_Images_OS

from monai.transforms import (
    Resize,
    ScaleIntensityRange,
    Transpose
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import wandb


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device="cuda"):
    #global run
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

            train_labels = train_labels.squeeze(1).long()
            loss = criterion(outputs, train_labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss     
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == train_labels).sum().item()
            total += train_labels.size(0)

            # Log metrics to wandb.
            #run.log({"val acc": correct, "val loss": loss})


            # Free memory
            del train_img_1, train_img_2, train_labels, outputs, preds
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f'Training complete. Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
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
    #global run

    all_preds = []
    all_ground_truths = []

    with torch.no_grad():
        for val_img_1, val_img_2, val_labels in val_loader:
            val_img_1, val_img_2, val_labels = val_img_1.to(device), val_img_2.to(device), val_labels.to(device)
            
            # Forward pass
            outputs = model(val_img_1, val_img_2)

            # Apply the mask to the loss
            loss = criterion(outputs, val_labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == val_labels).sum().item()
            total += val_labels.size(0)

            # Collect predictions and labels for F1 score
            all_preds.extend(preds.cpu().numpy())
            all_ground_truths.extend(val_labels.cpu().numpy())

            # Free memory
            #del val_img_1, val_img_2, val_labels, outputs, preds
            #torch.cuda.empty_cache()
            
    # Accuracy
    val_loss /= len(val_loader)
    val_acc = correct / total

    f1 = f1_score(all_labels, all_preds, average="weighted")  # Use "weighted" for class imbalance

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | Validation F1 score: {f1:.4f}")
    #show_roc_curve(masked_ground_truths, masked_probs)


    # ---------------------------
    # SAVE BEST MODEL
    # ---------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'./models/{model_name}.pth')
        print("Best model saved!")
   


# ---------------------------------------
# READ DATA
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/scratch/bmep/mfmakaske/training_scans/"
clinical_data_dir = "/scratch/bmep/mfmakaske/"

#data_dir = "L:/Basic/divi/jstoker/slicer_pdac/Master Students WS 24/Martijn/data/Training/paired_scans"
#clinical_data_dir = "C:/Users/P095550/Documents/CRLM-morph-features/CRLM-morph-features"

nifti_images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))   

# Create pairs (e.g., first and second file are paired)
image_pairs = [(nifti_images[i], nifti_images[i + 1]) for i in range(0, len(nifti_images) - 1, 2)]

pd_labels = pd.read_csv(os.path.join(clinical_data_dir, "training_labels_OS.csv"))
all_labels = torch.tensor(pd_labels.values.tolist()) #Fill in correct path. response, PFS, and OS

#-------------------------------------
# TRACK WITH WANDB
#-------------------------------------
"""
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="martijnmakaske-vrije-universiteit-amsterdam",
    # Set the wandb project where this run will be logged.
    project="CRLM-morph-features",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.001,
        "architecture": "Siames-Morph-CNN",
        "dataset": "CAIRO scaled images",
        "epochs": 1,
    },
)
"""

#------------------------------------
# INITIALIZE ENCODER
#------------------------------------

#resnet_model = torch.hub.load('Warvito/MedicalNet-models', 'medicalnet_resnet50')

# Remove the final classification layer (fc) to keep only the encoder part
#encoder = nn.Sequential(*list(resnet_model.children())[:-1])

# HYPERPARAMETERS
batch_size = 4
num_epochs = 100
model_name = "model_full_images_lr3"
class_weights = torch.tensor([1.6363636363636365, 0.496551724137931, 0.96, 3.0], dtype=torch.float32).to(device)

# Store metrics
best_val_acc = 0.0

# Perform a single train-test split with stratification
train_image_pairs, val_image_pairs, train_labels, val_labels = train_test_split(
    image_pairs, all_labels, test_size=0.2, random_state=42
)

# Change back to original size: (256, 256, 64)
# Create training and validation datasets
train_dataset = PairedMedicalDataset_Images(
    train_image_pairs, train_labels, transform=[ScaleIntensityRange(a_min=-200,
                                                                     a_max=400, b_min=0.0, b_max=1.0, clip=True), 
                                                Resize((256, 256, 64), 
                                                mode="trilinear"),
                                                Transpose((0, 3, 2, 1))]
)
val_dataset = PairedMedicalDataset_Images(
    val_image_pairs, val_labels, transform=[ScaleIntensityRange(a_min=-200,
                                                                 a_max=400, b_min=0.0, b_max=1.0, clip=True), 
                                            Resize((256, 256, 64), 
                                            mode="trilinear"),
                                            Transpose((0, 3, 2, 1))]
)


# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize the model
encoder = resnet.resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False, pretrained=True, shortcut_type="A", bias_downsample=True)
#add torch.compile
model = SiameseNetwork_Images_OS(encoder)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size= (num_epochs//3) , gamma=0.1)

# Train the model for this fold
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

#run.finish()
print("\nTraining and Validation Complete.")