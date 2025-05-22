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

# 1. Paths & constants
METADATA_CSV = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"
PATCHES_DIR = "/home/E19_FYP_Domain_Gen_Data/patches"
CONFIG_PATH = "../BioMedClip_Base_Eval/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "../BioMedClip_Base_Eval/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
NUM_WORKERS = 4

PROMPTS = {
    # Class 0 (e.g., "benign")
    0: {
        "positive": "Small mature lymphocytes centrally show no indication of active replication.",  
        "negative": "Large immature cells centrally demonstrate ongoing active replication"  
    },
    # Class 1 (e.g., "malignant")
    1: {
        "positive": "Large immature cells centrally demonstrate ongoing active replication",  
        "negative": "Small mature lymphocytes centrally show no indication of active replication."  
    }
}






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


class TextAnchorTripletDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, tokenizer, model):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model

        # Precompute text embeddings for ALL prompts
        self.text_embeddings = {}
        for lbl in [0, 1]:
            self.text_embeddings[lbl] = {
                "positive": self._encode_text(PROMPTS[lbl]["positive"]),
                "negative": self._encode_text(PROMPTS[lbl]["negative"])
            }

    def _encode_text(self, text: str) -> np.ndarray:
        """Helper to encode a text prompt."""
        tokens = self.tokenizer([text]).to(DEVICE)
        with torch.no_grad():
            emb = self.model.encode_text(tokens).cpu().numpy()[0]
        return emb

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor = self.features[idx]  # Image embedding
        label = self.labels[idx]     # 0 or 1

        # Positive: Text embedding of SAME class
        positive = self.text_embeddings[label]["positive"]

        # Negative: Text embedding of OPPOSITE class
        negative = self.text_embeddings[1 - label]["negative"]

        return (
            torch.from_numpy(anchor).float(),
            torch.from_numpy(positive).float(),
            torch.from_numpy(negative).float()
        )


# ------------------ Training Loop ------------------


def train_adapter(
    centers_features, centers_labels, device='cuda',
    lr=1e-3, batch_size=128, epochs=20,tokenizer=None, model=None
):
    # Prepare concatenated arrays
    feats = np.concatenate(centers_features)          # [N, D]
    labels = np.concatenate(centers_labels)           # [N]
    centers = np.concatenate(
        [[i] * len(f) for i, f in enumerate(centers_features)]
    )

    dataset = TextAnchorTripletDataset(features=feats, labels=labels,tokenizer=tokenizer,model=model)
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
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.6f}")

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

     # 2. Load metadata and filter center=0
    metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

     # Only use center=0 for training
    train_df = metadata_df[(metadata_df.split == 0) & (metadata_df.center == 0)].copy()  # <-- Only center 0
    center0_ds = BiomedCLIPDataset(train_df, preprocess)

    # Extract features ONLY for center 0
    print("Extracting features for center 0...")
    loader = DataLoader(center0_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    center0_features = []
    center0_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Center 0", unit="batch"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # Normalize

            center0_features.append(feats.cpu())
            center0_labels.append(labels.cpu())

    center0_features = torch.cat(center0_features).numpy()  # [N, D]
    center0_labels = torch.cat(center0_labels).numpy()      # [N]

    # Train adapter ONLY on center 0
    adapter = train_adapter(
        [center0_features],  # Pass as list to match original signature
        [center0_labels],
        device=DEVICE,
        lr=1e-3,
        batch_size=32,
        epochs=100,
        tokenizer=tokenizer,
        model=model
    )
    torch.save(adapter.state_dict(), 'adapter_center0_only_textEMB.pth')


if __name__ == "__main__":
    from prompt_learner import (
        BiomedCLIPDataset,
        append_filename_and_filepath,
    )
    main()
