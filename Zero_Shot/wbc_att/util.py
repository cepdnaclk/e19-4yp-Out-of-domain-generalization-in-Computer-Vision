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

# WBCAtt paths
WBCATT_IMAGE_BASE = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/"
WBCATT_TRAIN_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_train.csv"
WBCATT_VAL_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_val.csv"
WBCATT_TEST_CSV = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att/pbc_attr_v1_test.csv"

CONFIG_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CACHE_PATH = "cached"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


PromptPair = Tuple[str, str]
PromptSet = Tuple[str, ...]  # For multi-class prompts
InitialItem = Tuple[PromptSet, float]
PromptList = List[Tuple[PromptSet, float]]




class WBCAttDataset(Dataset):
    def __init__(self, csv_path, image_base, preprocess, binary_label: str = None):
        """
        WBCAtt Dataset for White Blood Cell classification.

        Args:
            csv_path: Path to the CSV file (train/val/test)
            image_base: Base directory containing the images
            preprocess: Image preprocessing function
        """
        self.df = pd.read_csv(csv_path)
        self.image_base = image_base
        self.preprocess = preprocess

        # Create label mapping from string labels to integers
        # Hardcoded label mapping
        if binary_label is None:
            self.label_to_idx = {
                "Basophil": 0,
                "Eosinophil": 1,
                "Lymphocyte": 2,
                "Monocyte": 3,
                "Neutrophil": 4
            }
        else:
            # Map the binary_label to 1, all others to 0
            self.label_to_idx = {
                label: (1 if label == binary_label else 0) for label in CLASSES}

        print(f"Label to index mapping (hardcoded): {self.label_to_idx}")
        self.idx_to_label = {idx: label for label,
                             idx in self.label_to_idx.items()}
        print(f"Index to label mapping: {self.idx_to_label}")

        # Convert labels to indices
        self.labels = [self.label_to_idx[label] for label in self.df['label']]
        print(f"Labels converted to indices: {self.labels}")

        # Construct full image paths
        self.image_paths = [os.path.join(
            image_base, row["path"]) for _, row in self.df.iterrows()]

        print(f"WBCAtt Dataset initialized with {len(self.df)} samples")
        print(f"Classes: {list(self.label_to_idx.keys())}")
        print(f"Label distribution:")
        for label, count in self.df['label'].value_counts().items():
            print(f"  {label}: {count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def get_num_classes(self):
        return len(self.label_to_idx)

    def get_class_names(self):
        return list(self.label_to_idx.keys())


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


def extract_embeddings(model, preprocess, split="train", cache_dir="./wbcatt_cache"):
    """
    Extract embeddings for WBCAtt dataset for a given split.
    Args:
        model: CLIP model
        preprocess: preprocessing function
        split: 'train', 'val', or 'test'
        cache_dir: directory to cache features/labels
    Returns:
        features_array: np.ndarray
        labels_array: np.ndarray
        dataset: WBCAttDataset instance (for accessing class information)
    """
    split_map = {
        "train": WBCATT_TRAIN_CSV,
        "val": WBCATT_VAL_CSV,
        "test": WBCATT_TEST_CSV
    }
    csv_path = split_map[split]

    os.makedirs(cache_dir, exist_ok=True)
    features_cache = os.path.join(cache_dir, f"wbcatt_{split}_features.npy")
    labels_cache = os.path.join(cache_dir, f"wbcatt_{split}_labels.npy")

    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        print(f"Loading cached embeddings for {split} split...")
        features_array = np.load(features_cache)
        labels_array = np.load(labels_cache)
        return features_array, labels_array

    # Create dataset and dataloader
    print(f"Creating WBCAtt dataset and dataloader for {split} split...")
    dataset = WBCAttDataset(csv_path, WBCATT_IMAGE_BASE, preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = model.to(DEVICE).eval()
    features, all_labels = [], []

    print(f"Extracting embeddings for {len(dataset)} {split} samples...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting WBCAtt {split} embeddings"):
            imgs = imgs.to(DEVICE)
            feats = model.encode_image(imgs)
            features.append(feats.cpu())
            all_labels.append(labels.cpu().numpy())

    features_array = torch.cat(features).numpy()
    labels_array = np.concatenate(all_labels)

    # Cache the results
    print(f"Caching embeddings to {features_cache} and {labels_cache}...")
    np.save(features_cache, features_array)
    np.save(labels_cache, labels_array)

    print(f"Extracted embeddings shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Unique labels: {np.unique(labels_array)}")

    return features_array, labels_array





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

        # Weighted cross entropy (for imbalanced classes)
        # Compute class weights: inverse frequency
        class_counts = np.bincount(y_true, minlength=len(prompt_set))
        class_weights = np.zeros(len(prompt_set), dtype=np.float32)
        for i, count in enumerate(class_counts):
            class_weights[i] = 1.0 / count if count > 0 else 0.0
        # Normalize weights to sum to num_classes
        class_weights = class_weights * len(prompt_set) / np.sum(class_weights)
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32).to(DEVICE)

        weighted_ce_loss = F.cross_entropy(
            logits, labels, weight=class_weights_tensor).item()
        inverted_weighted_ce = 1.0 / (1.0 + weighted_ce_loss)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': acc,
        'cm': cm,
        'report': report,
        'inverted_ce': inverted_ce,
        'inverted_weighted_ce': inverted_weighted_ce,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


