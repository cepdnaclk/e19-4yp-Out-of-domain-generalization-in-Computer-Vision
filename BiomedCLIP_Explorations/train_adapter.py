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
METADATA_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv"
PATCHES_DIR = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches"
CONFIG_PATH = "../BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "../BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CONTEXT_LENGTH = 256
BATCH_SIZE = 256
NUM_WORKERS = 8

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPTS = (
    'Small mature lymphocytes centrally show no indication of active replication.', 'Large immature cells centrally demonstrate ongoing active replication.'
)

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


class FeatureTripletDataset__AnchoredToPrompts(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, text_embeddings: np.ndarray):
        """
        features: np.array [N, D]
        labels:   np.array [N]
        text_embeddings: np.array [2, D] (neg_prompt, pos_prompt)
        """
        self.features = features
        self.labels = labels
        self.text_embeddings = text_embeddings

        # Precompute indices by label
        self.by_label = {
            0: np.where(labels == 0)[0],
            1: np.where(labels == 1)[0]
        }
        # total samples
        self.n = features.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Determine anchor class from the feature label at idx
        anchor_label = int(self.labels[idx])
        # Anchor is the corresponding text embedding
        anchor = self.text_embeddings[anchor_label]

        # Positive: a feature of the same class
        pos_indices = self.by_label[anchor_label]
        pos_idx = np.random.choice(pos_indices)
        positive = self.features[pos_idx]

        # Negative: a feature of the opposite class
        neg_label = 1 - anchor_label
        neg_indices = self.by_label[neg_label]
        neg_idx = np.random.choice(neg_indices)
        negative = self.features[neg_idx]

        # Convert to torch tensors
        return (
            torch.from_numpy(anchor).float(),
            torch.from_numpy(positive).float(),
            torch.from_numpy(negative).float()
        )


# ------------------ Training Loop ------------------


def train_adapter(
    dataset: torch.utils.data.Dataset,
    device: str = 'cuda',
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 20
) -> Adapter:
    """
    Train an Adapter using a triplet dataset of (anchor, positive, negative).

    Args:
        dataset: any Dataset that yields (anchor, positive, negative) triples.
        device: 'cuda' or 'cpu'.
        lr: learning rate.
        batch_size: batch size.
        epochs: number of epochs.

    Returns:
        The trained Adapter module.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=NUM_WORKERS)

    # Peek one batch to infer embedding dimension
    anchor_example, _, _ = next(iter(loader))
    dim = anchor_example.shape[1]

    adapter = Adapter(dim).to(device)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
        margin=0.2,
        distance_function=cosine_dist,
        reduction='mean'
    )
    optimizer = optim.Adam(adapter.parameters(), lr=lr)

    adapter.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            out_a = adapter(anchor)
            out_p = adapter(positive)
            out_n = adapter(negative)

            loss = triplet_loss_fn(out_a, out_p, out_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg = running_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — avg loss: {avg:.4f}")

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

    print("Device:", DEVICE)
    model.to(DEVICE).eval()

    random_state = 42
    # samples = 2048

    # 2. Load metadata and filter center=0
    metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    train_df = metadata_df[metadata_df.split == 0].copy()
    centers_ds = []
    for i in range(1):  # Change to 1 to use only center 0
        centers_ds.append(BiomedCLIPDataset(
            train_df[train_df.center == i], preprocess))

    # 3. Generate embeddings and store them for each center
    centers_features = [[] for _ in range(len(centers_ds))]
    centers_labels = [[] for _ in range(len(centers_ds))]

    for i, ds in enumerate(centers_ds):
        print(f"Extracting features for center {i}...")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2 if DEVICE.type == 'cuda' else 0)

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Center {i}", unit="batch"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                # labels = labels.to(DEVICE, non_blocking=True)

                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats.cpu())
                all_labels.append(labels)  # labels are already on CPU

        centers_features[i] = torch.cat(all_feats).numpy()
        centers_labels[i] = torch.cat(all_labels).numpy()

    # Prepare concatenated arrays
    all_feats = np.concatenate(centers_features)          # [N, D]
    all_labels = np.concatenate(centers_labels)           # [N]
    all_centers = np.concatenate(
        [[i] * len(f) for i, f in enumerate(centers_features)]
    )

    # generate text embeddings for prompts
    text_embeddings = []
    for prompt in PROMPTS:
        text = tokenizer(prompt, context_length=CONTEXT_LENGTH)
        text = text.to(DEVICE)
        with torch.no_grad():
            text_emb = model.encode_text(text)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_embeddings.append(text_emb.cpu().numpy())
    text_embeddings = np.stack(text_embeddings)  # [2, D]

    # all features, labels, centers, text should be in GPU memory
    all_feats = torch.from_numpy(all_feats).float().to(DEVICE)
    all_labels = torch.from_numpy(all_labels).long().to(DEVICE)
    all_centers = torch.from_numpy(all_centers).long().to(DEVICE)
    text_embeddings = torch.from_numpy(text_embeddings).float().to(DEVICE)

    # For the original center-to-center triplets:
    # base_dataset = FeatureTripletDataset(all_feats, all_labels, all_centers)
    # adapter = train_adapter(base_dataset, device=DEVICE, lr=1e-3, batch_size=32, epochs=100)

    # For the prompt-anchored triplets:
    epochs = 100
    prompt_dataset = FeatureTripletDataset__AnchoredToPrompts(
        all_feats, all_labels, text_embeddings)
    adapter = train_adapter(prompt_dataset, device=DEVICE,
                            lr=1e-3, batch_size=32, epochs=epochs)

    torch.save(adapter.state_dict(),
               f'adapter_weights_text_anchored_e{epochs}.pth')


if __name__ == "__main__":
    from prompt_learner import (
        BiomedCLIPDataset,
        append_filename_and_filepath,
    )
    main()
