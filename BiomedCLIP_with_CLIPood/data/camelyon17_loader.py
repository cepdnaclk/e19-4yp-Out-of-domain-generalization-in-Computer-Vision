import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CamelyonDataset(Dataset):
    def __init__(self, centers, metadata_path, root_dir, transform=None, split=None, include_test=False):
        """
        Custom Camelyon17 dataset loader.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
        self.centers = centers

        self.metadata = self.metadata[self.metadata['center'].isin(centers)]

        if split is not None:
            center_mask = self.metadata['center'].isin([0, 1, 2])
            if not include_test:
                self.metadata = self.metadata[(~center_mask) | (self.metadata['split'] == split)]

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

            if os.path.exists(img_path):
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
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CamelyonDataset(
        centers=[0, 1, 2],
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=train_transform,
        split=0
    )

    id_test_datasets = {
        f'id_test_center_{center}': CamelyonDataset(
            centers=[center],
            metadata_path=metadata_path,
            root_dir=data_root,
            transform=test_transform,
            split=1
        )
        for center in [0, 1, 2]
    }

    val_dataset = CamelyonDataset(
        centers=[3],
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=test_transform
    )

    test_dataset = CamelyonDataset(
        centers=[4],
        metadata_path=metadata_path,
        root_dir=data_root,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    id_test_loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for name, dataset in id_test_datasets.items()
    }

    return train_loader, val_loader, test_loader, id_test_loaders
