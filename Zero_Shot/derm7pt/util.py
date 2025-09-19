import heapq
import tokenize
import io
import random
from typing import Any, List, Optional, Set, Tuple
import re
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
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
from sklearn.metrics import f1_score

# 1. Paths & constants

# Derm7pt paths
DERM_META_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/meta.csv"
DERM_IMAGE_BASE = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/images/"
DERM_TRAIN_INDEXES = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/train_indexes.csv"
DERM_VAL_INDEXES = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/valid_indexes.csv"
DERM_TEST_INDEXES = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/test_indexes.csv"

CONFIG_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CACHE_PATH = "cached"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PromptPair = Tuple[str, str]
PromptSet = Tuple[str, ...]  # For multi-class prompts
InitialItem = Tuple[PromptPair, float]
PromptList = List[Tuple[PromptPair, float]]

class Derm7ptDataset(Dataset):
    def __init__(self, meta_csv, image_base, indexes_csv, preprocess, label_type):
        self.df = pd.read_csv(meta_csv)
        idx_df = pd.read_csv(indexes_csv)
        self.indexes = idx_df["indexes"].tolist()
        self.df = self.df.iloc[self.indexes].reset_index(drop=True)
        self.image_base = image_base
        self.preprocess = preprocess
        self.label_type = label_type

        def get_label(column, mapping, default=0):
            return self.df[column].map(mapping).fillna(default).astype(int).tolist()

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

            # Use the first image path column (clinic)
        self.image_paths = [os.path.join(
            image_base, row["derm"]) for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label



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


def extract_embeddings(model, preprocess, label_type, split="train", cache_dir="./derm7pt_cache"):
    """
    Extract embeddings for derm7pt dataset for a given label_type and split.
    Args:
        model: CLIP model
        preprocess: preprocessing function
        label_type: label_type indicating which label to use from the dataset: Eg: 'melanoma', 'pigment_network', etc.
        split: 'train', 'val', or 'test'
        cache_dir: directory to cache features/labels
    Returns:
        features_array: np.ndarray
        labels_array: np.ndarray
    """
    split_map = {"train": DERM_TRAIN_INDEXES,
                 "val": DERM_VAL_INDEXES, "test": DERM_TEST_INDEXES}
    indexes_csv = split_map[split]

    os.makedirs(cache_dir, exist_ok=True)
    features_cache = os.path.join(
        cache_dir, f"{split}_features_{label_type}.npy")
    labels_cache = os.path.join(cache_dir, f"{split}_labels_{label_type}.npy")
    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        features_array = np.load(features_cache)
        labels_array = np.load(labels_cache)
        return features_array, labels_array

    # Create dataset and dataloader
    print("Creating dataset and dataloader...")
    dataset = Derm7ptDataset(
        DERM_META_CSV, DERM_IMAGE_BASE, indexes_csv, preprocess, label_type)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = model.to(DEVICE).eval()
    features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting {split} embeddings for {label_type}"):
            imgs = imgs.to(DEVICE)
            feats = model.encode_image(imgs)
            features.append(feats.cpu())
            all_labels.append(labels.cpu().numpy())

    features_array = torch.cat(features).numpy()
    labels_array = np.concatenate(all_labels)
    np.save(features_cache, features_array)
    np.save(labels_cache, labels_array)

    return features_array, labels_array