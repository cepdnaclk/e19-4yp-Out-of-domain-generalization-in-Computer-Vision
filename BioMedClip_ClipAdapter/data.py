import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from open_clip import create_model_and_transforms
from tqdm import tqdm
import copy



# 1. Dataset Preparation
class CamelyonDataset(Dataset):
    def __init__(self, centers, metadata_path, root_dir, transform=None):
        """
        Args:
            centers (list): List of center indices to include (e.g., [0,1,2] for training)
            metadata_path (str): Path to metadata.csv
            root_dir (str): Root directory containing center_0 to center_4 folders
            transform (callable, optional): Optional transform to be applied
        """
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
        
        # Filter metadata for selected centers
        self.metadata = self.metadata[self.metadata['center'].isin(centers)]
        
        # Create mapping from image paths to labels
        self.samples = []
        for _, row in self.metadata.iterrows():
            patient = f"{int(row['patient']):03d}"
            node = int(row['node'])
            x_coord = int(row['x_coord'])
            y_coord = int(row['y_coord'])
            center = int(row['center'])
            label = int(row['tumor'])
            
            img_name = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
            img_path = os.path.join(root_dir, f"center_{center}", img_name)
            
            if os.path.exists(img_path):  # Only add if file exists
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(metadata_path, data_root, batch_size=32):
    # BiomedCLIP uses same normalization as CLIP
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation/Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Split centers: 3 train, 1 val, 1 test
    all_centers = [0, 1, 2, 3, 4]
    np.random.seed(42)  # For reproducibility
    # np.random.shuffle(all_centers)
    
    train_centers = all_centers[:3]
    val_centers = [all_centers[3]]
    test_centers = [all_centers[4]]
    
    print(f"Center split - Train: {train_centers}, Val: {val_centers}, Test: {test_centers}")
    
    # Create datasets
    train_dataset = CamelyonDataset(
        centers=train_centers,
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=train_transform
    )
    
    val_dataset = CamelyonDataset(
        centers=val_centers,
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=test_transform
    )
    
    test_dataset = CamelyonDataset(
        centers=test_centers,
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
