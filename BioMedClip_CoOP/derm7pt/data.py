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

class Derm7ptDataset(Dataset):
    def __init__(self, meta_csv, image_base, indexes_csv, transform=None, label_type="melanoma", few_shot=False, few_shot_no=2):
        """
        Args:
            meta_csv (str): Path to meta.csv file
            image_base (str): Base directory containing derm images
            indexes_csv (str): Path to train/val/test indexes csv file
            transform (callable, optional): Optional transform to be applied
            label_type (str): Type of label to use ('melanoma', 'pigment_network', etc.)
            few_shot (bool): Whether to use few-shot learning
            few_shot_no (int): Number of examples per class for few-shot
        """
        self.df = pd.read_csv(meta_csv)
        idx_df = pd.read_csv(indexes_csv)
        self.indexes = idx_df["indexes"].tolist()
        self.df = self.df.iloc[self.indexes].reset_index(drop=True)
        self.image_base = image_base
        self.transform = transform
        self.label_type = label_type

        def get_label(column, mapping, default=0):
            return self.df[column].map(mapping).fillna(default).astype(int).tolist()

        # Define label mappings for different tasks
        label_mappings = {
            "melanoma": lambda df: df["diagnosis"].str.contains("melanoma", case=False, na=False).astype(int).tolist(),
            "pigment_network": lambda df: get_label("pigment_network", {"absent": 0, "typical": 1, "atypical": 2}),
            "blue_whitish_veil": lambda df: get_label("blue_whitish_veil", {"present": 1, "absent": 0}),
            "vascular_structures": lambda df: get_label("vascular_structures", {
                "absent": 0, "arborizing": 0, "within regression": 0,
                "hairpin": 0, "comma": 0, "linear irregular": 1,
                "wreath": 0, "dotted": 1
            }),
            "pigmentation": lambda df: get_label("pigmentation", {"absent": 0, "diffuse regular": 0, "localized regular": 0, "diffuse irregular": 1, "localized irregular": 1}),
            "streaks": lambda df: get_label("streaks", {"absent": 0, "regular": 0, "irregular": 1}),
            "dots_and_globules": lambda df: get_label("dots_and_globules", {"absent": 0, "regular": 0, "irregular": 1}),
            "regression_structures": lambda df: get_label("regression_structures", {"absent": 0, "blue areas": 1, "white areas": 1, "combinations": 1}),
        }

        if self.label_type in label_mappings:
            self.labels = label_mappings[self.label_type](self.df)
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        # Use the derm image path column
        self.image_paths = [os.path.join(image_base, row["derm"]) for _, row in self.df.iterrows()]
        
        # Create samples list
        self.samples = []
        for i, (img_path, label) in enumerate(zip(self.image_paths, self.labels)):
            # Few-shot learning implementation
            if few_shot:
                label_counts = getattr(self, 'label_counts', {})
                if label not in label_counts:
                    label_counts[label] = 0
                if label_counts[label] >= few_shot_no:
                    continue
                label_counts[label] += 1
                self.label_counts = label_counts

            if os.path.exists(img_path):  # Only add if file exists
                self.samples.append((img_path, label))
                
        if few_shot:
            print(f"Number of samples in dataset: {len(self.samples)} for label_type: {self.label_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(meta_csv, image_base, train_indexes_csv, val_indexes_csv, test_indexes_csv, batch_size,few_shot,shots, label_type="melanoma"):
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
    
    # Create datasets according to the derm7pt splits:
    # - Train: train split from train_indexes.csv
    # - Val: val split from valid_indexes.csv  
    # - Test: test split from test_indexes.csv
    
    # Main training set 
    train_dataset = Derm7ptDataset(
        meta_csv=meta_csv,
        image_base=image_base,
        indexes_csv=train_indexes_csv,
        transform=train_transform,
        label_type=label_type,
        few_shot=few_shot,  # Few-shot learning enabled
        few_shot_no=shots
    )
    
    # Validation set 
    val_dataset = Derm7ptDataset(
        meta_csv=meta_csv,
        image_base=image_base,
        indexes_csv=val_indexes_csv,
        transform=test_transform,
        label_type=label_type,
        few_shot=few_shot,  # Few-shot learning enabled
        few_shot_no=shots
    )
    
    # Test set 
    test_dataset = Derm7ptDataset(
        meta_csv=meta_csv,
        image_base=image_base,
        indexes_csv=test_indexes_csv,
        transform=test_transform,
        label_type=label_type,
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