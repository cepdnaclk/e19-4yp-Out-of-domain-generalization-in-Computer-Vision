
from typing import Any, List, Optional, Set, Tuple
import re
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score
)
from typing import Callable, List, Tuple
import os
import pandas as pd
import numpy as np
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import json
from tqdm import tqdm
import time
from API_KEY import GEMINI_API_KEY
from google import genai
import ollama
import torch.nn.functional as F

# 1. Paths & constants
METADATA_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv"
PATCHES_DIR = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches"

CONFIG_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CACHE_PATH = "cached"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PromptPair = Tuple[str, str]
InitialItem = Tuple[PromptPair, float]
PromptList = List[Tuple[PromptPair, float]]
PromptSet = Tuple[str, ...]  # For multi-class prompts


class BiomedCLIPDataset(Dataset):
    def __init__(self, df, preprocess):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["tumor"].astype(int).tolist()
        self.preproc = preprocess

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
            img = self.preproc(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def append_filename_and_filepath(df):
    df["filename"] = df.apply(
        lambda r: f"patch_patient_{r.patient:03d}_node_{r.node}_x_{r.x_coord}_y_{r.y_coord}.png",
        axis=1
    )
    df["filepath"] = df.apply(
        lambda r: os.path.join(
            PATCHES_DIR,
            f"patient_{r.patient:03d}_node_{r.node}",
            r.filename
        ),
        axis=1
    )
    return df


def load_clip_model(
    config_path: str = CONFIG_PATH,
    model_name: str = MODEL_NAME,
    weights_path: str = WEIGHTS_PATH,
    device: torch.device = DEVICE,
) -> Tuple[torch.nn.Module, Callable]:
    """
    Load a CLIP model and its preprocessing pipeline from a JSON config.

    Args:
        config_path: Path to JSON config with model and preprocess settings.
        model_name: Name or HF hub identifier of the CLIP model.
        weights_path: Path to pretrained weights file.
        device: torch.device to place the model on.

    Returns:
        model: The loaded and eval()'d CLIP model on `device`.
        preprocess: A callable image preprocessing function.
    """
    # 1. Read configuration
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model_cfg, preproc_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]

    # 2. Register local config if needed
    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    # 3. Build tokenizer, model, and preprocess
    tokenizer = get_tokenizer(model_name)
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=weights_path,
        **{f"image_{k}": v for k, v in preproc_cfg.items()}
    )

    model = model.to(device).eval()
    return model, preprocess, tokenizer


def extract_center_embeddings(
    model: torch.nn.Module,
    preprocess: Callable,
    num_centers: int = 1,
    metadata_csv: str = METADATA_CSV,
    isTrain: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Reads metadata, splits into centers, and computes normalized image embeddings.

    Args:
        metadata_csv: Path to your metadata CSV.
        preprocess: Preprocessing function for images.
        model: A CLIP-style model with .encode_image().
        device: torch.device (e.g. torch.device("cuda")).
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader workers.
        num_centers: How many 'center' values you expect (default 4).

    Returns:
        centers_features: List of length `num_centers`, each a tensor of shape
                          [N_i, D] with normalized embeddings.
        centers_labels:   List of length `num_centers`, each a tensor of shape
                          [N_i] with corresponding labels.
    """
    # Load and annotate metadata
    metadata_df = pd.read_csv(metadata_csv, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    # Filter for training split 0: train, 1: test
    df = metadata_df[metadata_df.split == int(0 if isTrain else 1)]

    # Build datasets per center
    centers_ds = [
        BiomedCLIPDataset(df[df.center == i], preprocess)
        for i in range(num_centers)
    ]

    centers_features: List[torch.Tensor] = [[]
                                            for _ in range(num_centers)]
    centers_labels:   List[torch.Tensor] = [[]
                                            for _ in range(num_centers)]

    os.makedirs(f"{CACHE_PATH}/centers", exist_ok=True)
    for i, ds in enumerate(centers_ds):
        if isTrain:
            feature_path = f"{CACHE_PATH}/centers/train-center{i}_features.npy"
            label_path = f"{CACHE_PATH}/centers/train-center{i}_labels.npy"
        else:
            feature_path = f"{CACHE_PATH}/centers/test-center{i}_features.npy"
            label_path = f"{CACHE_PATH}/centers/test-center{i}_labels.npy"

        if os.path.exists(feature_path) and os.path.exists(label_path):
            print(f"Loading cached features for center {i}...")
            centers_features[i] = np.load(feature_path)
            centers_labels[i] = np.load(label_path)
            continue

        print(f"Extracting features for center {i}...")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Center {i}", unit="batch"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats.cpu())
                all_labels.append(labels)  # labels are already on CPU

        centers_features[i] = torch.cat(all_feats).numpy()
        centers_labels[i] = torch.cat(all_labels).numpy()

        np.save(feature_path, centers_features[i])
        np.save(label_path, centers_labels[i])

    return centers_features, centers_labels


def evaluate_prompt_set(
    prompt_set: PromptSet,
    image_feats: torch.Tensor,    # (N, D), precomputed
    image_labels: torch.Tensor,   # (N,)
    model,
    tokenizer
):
    # Encode all prompts (one per class)
    text_inputs = tokenizer(
        list(prompt_set),
        context_length=CONTEXT_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        text_feats = model.encode_text(
            text_inputs)           # (num_classes, D)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()

        feats = image_feats.to(DEVICE)                        # (N, D)
        labels = image_labels.to(DEVICE)                      # (N,)

        # Compute logits: (N, num_classes)
        logits = logit_scale * (feats @ text_feats.t())
        # (N, num_classes)
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)                          # (N,)

        y_pred = preds.cpu().numpy()
        y_true = labels.cpu().numpy()
        y_prob = probs.cpu().numpy()                          # (N, num_classes)

        # CrossEntropyLoss expects logits and integer labels
        ce_loss = F.cross_entropy(logits, labels).item()
        # Invert CE loss for scoring (lower loss → higher value)
        inverted_ce = 1.0 / (1.0 + ce_loss)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return {
        'accuracy': acc,
        'cm': cm,
        'report': report,
        'inverted_ce': inverted_ce,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }