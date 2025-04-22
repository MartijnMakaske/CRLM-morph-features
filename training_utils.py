# Imports
import numpy as np
import nibabel as nib
from monai.data import Dataset
from monai.transforms import (
    Compose
)
import torch
import torch.nn as nn


class PairedMedicalDataset_Images(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
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

        label = self.labels[idx]
        label = label.float()
 
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1.as_tensor(), img2.as_tensor(), label    
    
class PairedMedicalDataset_Full(Dataset):
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
        
        return img1.as_tensor(), img2.as_tensor(), metadata, label

class PairedMedicalDataset_Images_OS(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
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
        
        label = self.labels[idx]
        label = label.float()
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label
    
class SiameseNetwork_Images(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork_Images, self).__init__()
        self.base_model = base_model
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),     #use 2054 for resnet50, 1024 (+1) for resnet18
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,9)    # No softmax, because BCEWithLogitsLoss applies sigmoid internally
        )

    
    def forward(self, image1, image2):
        # Pass both inputs through the shared model
        output1 = self.base_model(image1)
        output2 = self.base_model(image2)

        # Apply adaptive average pooling to both outputs (resnet from monai flattens itself)
        #output1 = self.adaptive_pool(output1).view(output1.size(0), -1)  # Flatten after pooling
        #output2 = self.adaptive_pool(output2).view(output2.size(0), -1)  # Flatten after pooling

        combined_embeddings = torch.cat((output1, output2), dim=1)
        output3 = self.classifier(combined_embeddings)
        return output3

class SiameseNetwork_Full(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork_Full, self).__init__()
        self.base_model = base_model
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),     #use 2054 for resnet50, 1024 (+1) for resnet18
            nn.LeakyReLU(),
            nn.Linear(518,256),     #use 2054 for resnet50
            nn.LeakyReLU(),
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

class SiameseNetwork_Images_OS(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork_Images_OS, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),     #use 2054 for resnet50, 1024 (+1) for resnet18
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,4)   
        )

    
    def forward(self, image1, image2):
        # Pass both inputs through the shared model
        output1 = self.base_model(image1)
        output2 = self.base_model(image2)

        combined_embeddings = torch.cat((output1, output2), dim=1)
        output3 = self.classifier(combined_embeddings)
        return output3