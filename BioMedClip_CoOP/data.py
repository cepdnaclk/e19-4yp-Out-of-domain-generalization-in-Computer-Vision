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

class CamelyonDataset(Dataset):
    def __init__(self, centers, metadata_path, root_dir, transform=None, split=None, include_test=False,few_shot=False,few_shot_no=2):
        """
        Args:
            centers (list): List of center indices to include (e.g., [0,1,2])
            metadata_path (str): Path to metadata.csv
            root_dir (str): Root directory containing center_0 to center_4 folders
            transform (callable, optional): Optional transform to be applied
            split (int, optional): 0 for train, 1 for test (only for centers 0-2)
            include_test (bool): Whether to include test split for centers (default False)
        """
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
        self.centers = centers
        
        # Filter metadata for selected centers
        self.metadata = self.metadata[self.metadata['center'].isin(centers)]
        
        # For centers 0-2, we can select train or test splits
        if split is not None:
            center_mask = self.metadata['center'].isin([0, 1, 2])
            if not include_test:
                # Only get the specified split for centers 0-2
                self.metadata = self.metadata[(~center_mask) | (self.metadata['split'] == split)]
            else:
                # Get all data from centers 0-2 regardless of split
                pass
        
        # Create mapping from image paths to labels
        self.samples = []
        for _, row in self.metadata.iterrows():
            # take first two tumor and non-tumor images only
            # Added by Mansitha Few Shot
            if few_shot:
                tumor_counts = getattr(self, 'tumor_counts', {0: 0, 1: 0})
                label = int(row['tumor'])
                if tumor_counts[label] >= few_shot_no:
                    continue
                tumor_counts[label] += 1
                self.tumor_counts = tumor_counts

            patient = f"{int(row['patient']):03d}"
            node = int(row['node'])
            x_coord = int(row['x_coord'])
            y_coord = int(row['y_coord'])
            center = int(row['center'])
            label = int(row['tumor'])
            
            img_name = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
            img_path = os.path.join(root_dir, f"patient_{patient}_node_{node}", img_name)
            if os.path.exists(img_path):  # Only add if file exists
                self.samples.append((img_path, label))
        if few_shot:
            print(f"Number of samples in dataset: {len(self.samples)} and those are {self.samples}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(metadata_path, data_root, batch_size):
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
    
    # Create datasets according to the specified splits:
    # - Train: train splits from centers 0,1,2 (split=0)
    # - ID Test: test splits from centers 0,1,2 (split=1)
    # - Val: all of center 3 (no split)
    # - Test: all of center 4 (no split)
    
    # Main training set (train splits from centers 0,1,2)
    train_dataset = CamelyonDataset(
        centers=[0],
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=train_transform,
        split=0 , # train split,
        few_shot=True,  # Few-shot learning enabled
        few_shot_no=8
    )
    
    # In-distribution test sets (test splits from centers 0,1,2)
    id_test_datasets = {
        f'id_test_center_{center}': CamelyonDataset(
            centers=[center],
            metadata_path=metadata_path,
            root_dir=data_root,
            transform=test_transform,
            split=1  # test split
        )
        for center in [0, 1, 2]
    }
    
    # Validation set (all of center 3)
    val_dataset = CamelyonDataset(
        centers=[3],
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=test_transform,
        few_shot=True,  # Few-shot learning enabled
        few_shot_no=16

    )
    
    # Test set (all of center 4)
    test_dataset = CamelyonDataset(
        centers=[4],
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
    
    # Create in-distribution test loaders
    id_test_loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        for name, dataset in id_test_datasets.items()
    }
    
    return train_loader, val_loader, test_loader, id_test_loaders