import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
# from open_clip import create_model_and_transforms
from tqdm import tqdm
import copy


class WBCAttDataset(Dataset):
    def __init__(self, csv_path, image_base, transform=None, label_col="label", few_shot=False, few_shot_no=2):
        """
        WBCAtt dataset loader.
        Args:
            csv_path (str): Path to wbcatt train/val/test csv file
            image_base (str): Base directory containing wbcatt images
            transform (callable, optional): Optional transform to be applied
            label_col (str): Column name for label in csv
        """
        self.df = pd.read_csv(csv_path)
        self.image_base = image_base
        self.transform = transform
        self.label_col = label_col
        self.few_shot = few_shot
        self.few_shot_no = few_shot_no

        # Create label mapping from string labels to integer indices
        unique_labels = sorted(self.df[label_col].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        # Convert labels to indices
        self.labels = [self.label_to_idx[label] for label in self.df[label_col]]
        # Use 'path' column for image paths
        self.image_paths = [os.path.join(image_base, row["path"]) for _, row in self.df.iterrows()]

        # Few-shot logic: select only N samples per class if enabled
        self.samples = []
        if self.few_shot:
            label_counts = {}
            for img_path, label in zip(self.image_paths, self.labels):
                if label not in label_counts:
                    label_counts[label] = 0
                if label_counts[label] < self.few_shot_no:
                    if os.path.exists(img_path):
                        self.samples.append((img_path, label))
                        label_counts[label] += 1
            print(f"Few-shot enabled: {self.few_shot_no} samples per class. Total: {len(self.samples)}")
        else:
            for img_path, label in zip(self.image_paths, self.labels):
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_csv, val_csv, test_csv, image_base, batch_size, label_col="label", few_shot=False, few_shot_no=2):
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
    
    # Create datasets according to the wbcatt splits:
    # - Train: train split from train_csv
    # - Val: val split from val_csv
    # - Test: test split from test_csv
    

    # Main training set
    train_dataset = WBCAttDataset(
        csv_path=train_csv,
        image_base=image_base,
        transform=train_transform,
        label_col=label_col,
        few_shot=few_shot,
        few_shot_no=few_shot_no
    )

    # Validation set
    val_dataset = WBCAttDataset(
        csv_path=val_csv,
        image_base=image_base,
        transform=test_transform,
        label_col=label_col,
        few_shot=False,
        few_shot_no=2
    )

    # Test set
    test_dataset = WBCAttDataset(
        csv_path=test_csv,
        image_base=image_base,
        transform=test_transform,
        label_col=label_col,
        few_shot=False,
        few_shot_no=2
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