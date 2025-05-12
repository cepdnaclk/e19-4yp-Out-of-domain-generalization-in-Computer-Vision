import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from prompt_learner import (
    BiomedCLIPDataset,
    append_filename_and_filepath,
)
# 1. Paths & constants
METADATA_CSV = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"
PATCHES_DIR = "/home/E19_FYP_Domain_Gen_Data/patches"
CONFIG_PATH = "../BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "../BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
NUM_WORKERS = 4

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ Adapter Definition ------------------
class Adapter(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

# ------------------ Triplet Dataset ------------------


def cosine_dist(x, y):
    # returns 1 − cosine_similarity, so smaller means more similar
    return 1.0 - F.cosine_similarity(x, y, dim=1)


class FeatureTripletDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, centers: np.ndarray):
        """
        features: np.array [N, D]
        labels:   np.array [N]
        centers:  np.array [N]  # e.g. 0,1,2 indicating domain
        """
        self.features = features
        self.labels = labels
        self.centers = centers

        # List of all unique centers
        self.unique_centers = list(np.unique(centers))

        # Precompute indices by (center, label)
        self.by_center_label = {}
        for c in self.unique_centers:
            for lbl in [0, 1]:
                mask = (centers == c) & (labels == lbl)
                self.by_center_label[(c, lbl)] = np.where(mask)[0]

        # We'll allow any sample to be an anchor
        self.all_idxs = np.arange(len(labels))

    def __len__(self):
        return len(self.all_idxs)

    def __getitem__(self, idx):
        # 1) Anchor from any center
        a_idx = self.all_idxs[idx]
        anchor = self.features[a_idx]
        a_lbl = self.labels[a_idx]
        a_center = self.centers[a_idx]

        # 2) Choose a different center j != a_center
        other_centers = [c for c in self.unique_centers if c != a_center]
        pos_center = np.random.choice(other_centers)

        # 3) Positive: same class as anchor, from pos_center
        pos_idx = np.random.choice(self.by_center_label[(pos_center, a_lbl)])
        positive = self.features[pos_idx]

        # 4) Negative: opposite class from the same pos_center
        neg_lbl = 1 - a_lbl
        neg_idx = np.random.choice(self.by_center_label[(pos_center, neg_lbl)])
        negative = self.features[neg_idx]

        # Return tensors
        return (
            torch.from_numpy(anchor).float(),
            torch.from_numpy(positive).float(),
            torch.from_numpy(negative).float()
        )


# ------------------ Training Loop ------------------


def train_adapter(
    centers_features, centers_labels, device='cuda',
    lr=1e-3, batch_size=128, epochs=20
):
    # Prepare concatenated arrays
    feats = np.concatenate(centers_features)          # [N, D]
    labels = np.concatenate(centers_labels)           # [N]
    centers = np.concatenate(
        [[i] * len(f) for i, f in enumerate(centers_features)]
    )

    dataset = FeatureTripletDataset(feats, labels, centers)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    dim = feats.shape[1]
    adapter = Adapter(dim).to(device)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        margin=0.2, distance_function=cosine_dist, reduction='mean')
    # triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(adapter.parameters(), lr=lr)

    adapter.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass through adapter
            out_a = adapter(anchor)
            out_p = adapter(positive)
            out_n = adapter(negative)

            # Triplet loss: align same-class across domains
            L_trip = triplet_loss_fn(out_a, out_p, out_n)

            # Euclidean loss: preserve original center-0 content
            # L_euc  = cosine_dist(out_a, anchor).mean()

            # Combine losses
            loss = L_trip
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    return adapter


def main():
    # 1. Load model and preprocess
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    model_cfg, preproc_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]

    # register local config if needed
    if (not MODEL_NAME.startswith(HF_HUB_PREFIX)
            and MODEL_NAME not in _MODEL_CONFIGS):
        _MODEL_CONFIGS[MODEL_NAME] = model_cfg

    tokenizer = get_tokenizer(MODEL_NAME)
    model, _, preprocess = create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=WEIGHTS_PATH,
        **{f"image_{k}": v for k, v in preproc_cfg.items()}
    )

    model.to(DEVICE).eval()

    random_state = 42
    # samples = 2048

    # 2. Load metadata and filter center=0
    metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    train_df = metadata_df[metadata_df.split == 0].copy()
    centers_ds = []
    for i in range(3):
        centers_ds.append(BiomedCLIPDataset(
            train_df[train_df.center == i], preprocess))

    # 3. Generate embeddings and store them for each center
    centers_features = [[] for _ in range(len(centers_ds))]
    centers_labels = [[] for _ in range(len(centers_ds))]

    for i, ds in enumerate(centers_ds):
        print(f"Extracting features for center {i}...")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Center {i}", unit="batch"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())

        centers_features[i] = torch.cat(all_feats).numpy()
        centers_labels[i] = torch.cat(all_labels).numpy()

    # train and save the adapter
    adapter = train_adapter(centers_features[0:3], centers_labels[0:3], device=DEVICE,
                            lr=1e-3, batch_size=32, epochs=50)
    torch.save(adapter.state_dict(), 'adapter_weights.pth')


if __name__ == "__main__":
    main()
