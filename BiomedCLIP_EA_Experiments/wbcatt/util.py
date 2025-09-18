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
from API_KEY import GEMINI_API_KEY, CHATGPT_ENDPOINT, CHATGPT_API_KEY
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


class KneePointAnalysis:
    """
    A utility class to perform knee-point (elbow) analysis on a collection of scores,
    typically obtained from a PriorityQueue.
    """

    def __init__(self, scores: List[float]):
        """
        Initializes the analyzer with a list of scores.
        Scores are expected to be in descending order for knee-point analysis.
        Ensures scores are numeric by casting to float.
        """
        # Ensure all scores are floats before sorting and storing
        self._scores = sorted([float(s) for s in scores], reverse=True)

    def find_knee_point(self, k_range: Optional[List[int]] = None) -> int:
        """
        Finds the knee point (elbow) in the sorted scores using the 'distance from line' method.
        This suggests a natural 'n' for selecting the best solutions.

        Args:
            k_range: An optional list of K values (number of solutions/ranks) to consider for analysis.
                     If None, it considers all possible 'n' up to the number of scores.
        Returns:
            The recommended number of 'n' (solutions) based on the knee point.
            Returns 0 if there are no scores or only one score.
        """
        if len(self._scores) <= 1:
            return len(self._scores)

        # Prepare x-values (ranks) and y-values (scores) for the full dataset
        x_values_all = np.arange(1, len(self._scores) + 1)
        # This conversion now should be safe as _scores are floats
        y_values_all = np.array(self._scores)

        # Determine the subset of data for analysis if k_range is provided
        if k_range is None:
            x_values_for_analysis = x_values_all
            y_values_for_analysis = y_values_all
        else:
            # Filter x_values and y_values based on k_range
            valid_indices = [i for i, k in enumerate(
                x_values_all) if k in k_range and k <= len(self._scores)]
            if not valid_indices:
                print(
                    "Warning: No valid ranks in k_range for the current data size. Returning 0.")
                return 0
            x_values_for_analysis = x_values_all[valid_indices]
            y_values_for_analysis = y_values_all[valid_indices]

        # Handle cases with insufficient points for line drawing
        if len(x_values_for_analysis) < 2:
            return len(x_values_for_analysis) if len(x_values_for_analysis) > 0 else 0

        # If the analysis range is just two points, the knee is trivially the first point,
        # or the second if the first is disproportionately high. For robustness, if only 2 points,
        # the 'knee' concept is weak.
        if len(x_values_for_analysis) == 2:
            return x_values_for_analysis[0]

        # Calculate the line between the first and last point of the *analysis subset*
        p1 = (x_values_for_analysis[0], y_values_for_analysis[0])
        p2 = (x_values_for_analysis[-1], y_values_for_analysis[-1])

        # Calculate line equation: y = mx + c  =>  mx - y + c = 0
        # Handle the edge case where x_values_for_analysis are all the same (vertical line)
        if (p2[0] - p1[0]) == 0:
            return x_values_for_analysis[0]  # Fallback to first point

        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]

        A, B, C = m, -1, c  # Coefficients for Ax + By + C = 0

        distances = []
        for i in range(len(x_values_for_analysis)):
            x0, y0 = x_values_for_analysis[i], y_values_for_analysis[i]
            dist = np.abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
            distances.append(dist)

        # The knee point is where the distance is maximized
        knee_index = np.argmax(distances)
        recommended_n = x_values_for_analysis[knee_index]  # The rank itself

        return recommended_n


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


def extract_embeddings_for_binary_classification(model, preprocess, binary_label, split="train", cache_dir="./wbcatt_cache"):
    """
    Extract embeddings for WBCAtt dataset for binary classification of a specific cell type.
    Args:
        model: CLIP model
        preprocess: preprocessing function
        binary_label: The cell type to classify as positive (e.g., "Basophil", "Eosinophil", etc.)
                     All other cell types will be labeled as negative (0)
        split: 'train', 'val', or 'test'
        cache_dir: directory to cache features/labels
    Returns:
        features_array: np.ndarray
        labels_array: np.ndarray (binary: 0 for other classes, 1 for binary_label)
    """
    split_map = {
        "train": WBCATT_TRAIN_CSV,
        "val": WBCATT_VAL_CSV,
        "test": WBCATT_TEST_CSV
    }
    csv_path = split_map[split]

    os.makedirs(cache_dir, exist_ok=True)
    # Create binary-label-specific cache files
    features_cache = os.path.join(
        cache_dir, f"wbcatt_{split}_{binary_label}_binary_features.npy")
    labels_cache = os.path.join(
        cache_dir, f"wbcatt_{split}_{binary_label}_binary_labels.npy")

    if os.path.exists(features_cache) and os.path.exists(labels_cache):
        print(
            f"Loading cached binary embeddings for {split} split (binary_label: {binary_label})...")
        features_array = np.load(features_cache)
        labels_array = np.load(labels_cache)
        return features_array, labels_array

    # Create dataset and dataloader with binary classification
    print(
        f"Creating WBCAtt binary dataset and dataloader for {split} split (binary_label: {binary_label})...")
    dataset = WBCAttDataset(csv_path, WBCATT_IMAGE_BASE,
                            preprocess, binary_label=binary_label)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = model.to(DEVICE).eval()
    features, all_labels = [], []

    print(
        f"Extracting binary embeddings for {len(dataset)} {split} samples (binary_label: {binary_label})...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Extracting WBCAtt {split} binary embeddings ({binary_label})"):
            imgs = imgs.to(DEVICE)
            feats = model.encode_image(imgs)
            features.append(feats.cpu())
            all_labels.append(labels.cpu().numpy())

    features_array = torch.cat(features).numpy()
    labels_array = np.concatenate(all_labels)

    # Cache the results
    print(
        f"Caching binary embeddings to {features_cache} and {labels_cache}...")
    np.save(features_cache, features_array)
    np.save(labels_cache, labels_array)

    print(f"Extracted binary embeddings shape: {features_array.shape}")
    print(f"Binary labels shape: {labels_array.shape}")
    print(
        f"Binary label distribution: {np.bincount(labels_array)} (0: other classes, 1: {binary_label})")

    return features_array, labels_array


def evaluate_prompt_list(
    prompt_list: PromptList,
    image_feats: torch.Tensor,  # (N, D), precomputed
    image_labels: torch.Tensor,  # (N,)
    model,
    tokenizer,
    unweighted: bool = False  # If True, all prompts are treated equally
):
    all_weighted_probs = []
    total_weight = 0.0

    # Ensure image feats and labels are on the correct device once
    print(f"shape of image_feats:dtype: {image_feats} ")
    feats = image_feats.to(DEVICE)
    labels = image_labels.to(DEVICE)

    with torch.no_grad():
        for prompt_set, original_weight in prompt_list:
            # Determine the effective weight
            effective_weight = 1.0 if unweighted else original_weight

            # Skip if effective_weight is zero to avoid unnecessary computations
            if effective_weight == 0 and not unweighted:
                continue

            text_inputs = tokenizer(list(prompt_set),
                                    context_length=CONTEXT_LENGTH
                                    ).to(DEVICE)

            text_feats = model.encode_text(text_inputs)  # (2, D)
            text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
            logit_scale = model.logit_scale.exp()

            # compute all logits at once: (N, 2)
            logits = logit_scale * (feats @ text_feats.t())
            probs = logits.softmax(dim=1)

            # For multiclass: collect weighted probabilities for all classes
            current_probs = probs.cpu().numpy()  # shape: (N, num_classes)
            all_weighted_probs.append(current_probs * effective_weight)
            total_weight += effective_weight

    # Perform ensemble
    if total_weight == 0:
        # If all weights were zero, or prompt_list was empty and unweighted was False
        # This case handles scenarios where no valid prompts or non-zero weights were found.
        # It might be better to return an error or specific empty results depending on desired behavior.
        # For now, let's return a dict with NaN or 0 for metrics for clarity,
        # or raise an error if this state indicates a problem.
        raise ValueError(
            "Total effective weight of prompts is zero. Cannot perform ensemble.")

    # Convert list of arrays to a single array and sum along the first dimension
    ensemble_probs_unnormalized = np.sum(all_weighted_probs, axis=0)
    ensemble_probs = ensemble_probs_unnormalized / total_weight
    # Convert list of arrays to a single array and sum along the first dimension
    # For multiclass: shape (N, num_classes)
    ensemble_probs_unnormalized = np.sum(all_weighted_probs, axis=0)
    ensemble_probs = ensemble_probs_unnormalized / total_weight

    # Convert probabilities to predictions (multiclass: argmax)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    y_true = labels.cpu().numpy()

    # metrics
    acc = accuracy_score(y_true, ensemble_preds)
    cm = confusion_matrix(y_true, ensemble_preds)
    report = classification_report(
        y_true, ensemble_preds, digits=4, zero_division=0)

    return {'accuracy': acc, 'cm': cm, 'report': report}


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


def _force_double_quotes(code: str) -> str:
    """
    Rewrites every Python string-literal in `code` to use double-quotes,
    properly handling apostrophes and other special characters.
    """
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    new_tokens = []
    for toknum, tokval, start, end, line in tokens:
        if toknum == tokenize.STRING:
            # Get the actual string value
            value = ast.literal_eval(tokval)

            # Create a new string literal with double quotes
            # Properly escape any double quotes or backslashes in the string
            # This automatically handles escaping correctly
            tokval = json.dumps(value)

        new_tokens.append((toknum, tokval))
    return tokenize.untokenize(new_tokens)


def extract_and_parse_prompt_list_of_templates(code: str) -> List[str]:
    """
    From a string of Python code, finds the first occurrence of
        = [ ... ]
    and parses that bracketed literal into a List[str].

    Raises:
        ValueError if no list literal is found or it’s malformed.
    """
    import re
    import ast

    # 1) grab everything from the first '=' up to the matching ']'
    m = re.search(r'=\s*(\[\s*[\s\S]*?\])', code)
    if not m:
        raise ValueError("No list literal found after an '=' in the code")
    list_str = m.group(1)

    # 2) safely evaluate it (only literals)
    try:
        data = ast.literal_eval(list_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Malformed list literal: {e}")

    # 3) validate shape
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Parsed object is not a list of strings")

    prompt_templates = [str(x) for x in data]
    prompts = []

    for template in prompt_templates:
        prompt_set = tuple(template.replace("<cell_type>", cls)
                           for cls in CLASSES)
        prompts.append(prompt_set)
    return prompts


def extract_and_parse_prompt_list(code: str) -> List[Tuple[str, ...]]:
    """
    From a string of Python code, finds the first occurrence of
        = [ ... ]
    and parses that bracketed literal into a List[Tuple[str, ...]].

    Raises:
        ValueError if no list literal is found or it’s malformed.
    """
    # 1) grab everything from the first '=' up to the matching ']'
    m = re.search(r'=\s*(\[\s*[\s\S]*?\])', code)
    if not m:
        raise ValueError("No list literal found after an '=' in the code")
    list_str = m.group(1)

    # 2) safely evaluate it (only literals)
    try:
        data: Any = ast.literal_eval(list_str)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Malformed list literal: {e}")

    # 3) validate shape
    if not isinstance(data, list) or not all(
        isinstance(item, (list, tuple)) and all(isinstance(x, str) for x in item) for item in data
    ):
        raise ValueError(
            "Parsed object is not a list of tuples/lists of strings"
        )

    # 4) convert to List[Tuple[str, ...]]
    return [tuple(str(x) for x in item) for item in data]


def extract_and_parse_prompt_list_with_scores(code: str) -> List[Tuple[str, str, float]]:
    """
    From a string of Python code, finds the first occurrence of
        prompts = [ ... ]
    and parses that bracketed literal into a List[Tuple[str, str, float]].
    Expects format: ("neg", "pos"), score,

    Raises:
        ValueError if no list literal is found or it's malformed.
    """
    # 1) Find the prompts list declaration
    m = re.search(r'prompts\s*=\s*(\[\s*[\s\S]*?\])', code)
    if not m:
        raise ValueError("No 'prompts = [...]' list literal found in the code")
    list_str = m.group(1)

    # 2) Clean and normalize the string
    # Remove newlines and extra spaces for easier parsing
    cleaned = ' '.join(list_str.split())
    # Ensure we have proper comma separation between items
    cleaned = cleaned.replace('),', '), ')

    # 3) Split into individual items while preserving structure
    items = []
    current_item = []
    depth = 0  # track nesting level for tuples
    buffer = ""

    for char in cleaned[1:-1]:  # skip outer brackets
        if char == '(':
            depth += 1
            buffer += char
        elif char == ')':
            depth -= 1
            buffer += char
        elif char == ',' and depth == 0:
            # Only split on top-level commas
            if buffer.strip():
                current_item.append(buffer.strip())
                buffer = ""
            if len(current_item) == 2:  # we have both tuple and score
                items.append(tuple(current_item))
                current_item = []
        else:
            buffer += char

    # Add the last item if any
    if buffer.strip():
        current_item.append(buffer.strip())
    if current_item:
        items.append(tuple(current_item))

    # 4) Parse each item into (neg, pos, score)
    parsed_items = []
    for item in items:
        if len(item) != 2:
            raise ValueError(f"Expected tuple and score, got: {item}")

        # Parse the prompt tuple
        try:
            prompt_tuple = ast.literal_eval(item[0])
            if not isinstance(prompt_tuple, tuple) or len(prompt_tuple) != 2:
                raise ValueError(
                    f"Expected 2-element tuple, got: {prompt_tuple}")
            neg, pos = prompt_tuple
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Malformed prompt tuple: {e}")

        # Parse the score
        try:
            score = float(item[1])
        except ValueError as e:
            raise ValueError(f"Malformed score value: {e}")

        parsed_items.append((str(neg), str(pos), score))

    return parsed_items


def extract_and_parse_prompt_tuple(code: str) -> Tuple[str, ...]:
    """
    From a string of Python code, finds the first literal tuple of strings
    (e.g. ("neg prompt","pos prompt", ...)) and returns it as Tuple[str, ...].

    Raises:
        ValueError if no suitable tuple of strings is found.
    """
    # Parse into an AST
    tree = ast.parse(code)

    # Walk the tree looking for a Tuple node with all string constants
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            for elt in node.elts
        ):
            return tuple(elt.value for elt in node.elts)

    raise ValueError("No tuple of strings found in code")


def _fix_quote_issues(code: str) -> str:
    """
    Fix common malformed string literal issues inside the first list literal
    that follows an '=' in the provided code string.

    Approach:
      - Find the first assignment to a list: locate the '[' that starts the list and its matching ']'.
      - For each tuple inside that list, identify string elements and normalize them:
        - Protect apostrophes inside words (cell's) so they are not misinterpreted.
        - Convert inner single-quoted phrases (e.g. 'ground glass') to double-quoted phrases.
        - Emit a safe Python string literal using json.dumps(...) (double-quoted, properly escaped).
      - Rebuild and return corrected code (only the list region is changed).
    """
    # find the bracketed list following the first '='
    eq_m = re.search(r'=\s*\[', code)
    if not eq_m:
        # nothing to fix
        return code

    list_start = code.find('[', eq_m.start())
    if list_start == -1:
        return code

    # find matching closing ']' for the list (simple bracket counter)
    depth = 0
    list_end = None
    for i in range(list_start, len(code)):
        c = code[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                list_end = i
                break
    if list_end is None:
        # can't find end bracket: give up and return original
        return code

    list_region = code[list_start:list_end + 1]

    def fix_tuple_region(tuple_region: str) -> str:
        """
        tuple_region contains '(' ... ')' inclusive. We fix each element string inside.
        """
        assert tuple_region.startswith('(') and tuple_region.endswith(')')
        inner = tuple_region[1:-1]
        out_parts: List[str] = ['(']
        pos = 0
        L = len(inner)

        while pos < L:
            # copy leading whitespace
            ws_match = re.match(r'\s*', inner[pos:])
            if ws_match:
                ws = ws_match.group(0)
                out_parts.append(ws)
                pos += len(ws)

            if pos >= L:
                break

            ch = inner[pos]
            if ch in ("'", '"'):
                # heuristically find the closing quote that is followed only by optional whitespace
                # and then comma or end-of-tuple
                quote = ch
                j = pos + 1
                found_closing = False
                while j < L:
                    if inner[j] == '\\':
                        # skip escaped char
                        j += 2
                        continue
                    if inner[j] == quote:
                        # lookahead
                        k = j + 1
                        while k < L and inner[k].isspace():
                            k += 1
                        if k >= L or inner[k] in (',',):
                            found_closing = True
                            break
                    j += 1

                if not found_closing:
                    # fallback: try to find a closing quote before the next top-level comma
                    next_comma = inner.find(',', pos)
                    if next_comma == -1:
                        # take the rest
                        j = L - 1
                    else:
                        # take up to just before the comma
                        j = next_comma - 1
                        # ensure j is within bounds
                        if j < pos:
                            j = pos

                # element raw includes the quotes (pos .. j)
                element_raw = inner[pos:j+1]
                # find if there's trailing whitespace and a comma immediately after j
                k = j + 1
                trailing_ws = ''
                while k < L and inner[k].isspace():
                    trailing_ws += inner[k]
                    k += 1
                trailing_comma = ''
                if k < L and inner[k] == ',':
                    trailing_comma = ','
                    k += 1

                # original inner content (without the outer quotes)
                orig_inner = element_raw[1:-1]

                # --- transform inner text ---
                temp = orig_inner

                # protect word-internal apostrophes so they are not confused with quote-pairs
                temp = re.sub(r"(?<=\w)'(?=\w)", "__APOST__", temp)

                # convert inner single-quoted phrases to double-quoted phrases
                # (only acts on remaining single-quote pairs)
                temp = re.sub(r"'([^']*?)'", r'"\1"', temp)

                # restore protected apostrophes
                temp = temp.replace("__APOST__", "'")

                # emit a safe Python literal using json.dumps (double-quoted and escaped)
                safe_literal = json.dumps(temp)

                out_parts.append(safe_literal)
                if trailing_ws:
                    out_parts.append(trailing_ws)
                if trailing_comma:
                    out_parts.append(trailing_comma)
                pos = k
            else:
                # non-quoted token (e.g., stray tokens) -- copy until next comma
                next_comma = inner.find(',', pos)
                if next_comma == -1:
                    out_parts.append(inner[pos:])
                    pos = L
                else:
                    out_parts.append(inner[pos:next_comma])
                    out_parts.append(',')
                    pos = next_comma + 1

        out_parts.append(')')
        return ''.join(out_parts)

    # Walk the list_region and replace each top-level tuple region with its fixed version.
    fixed_list_builder: List[str] = []
    i = 0
    N = len(list_region)
    while i < N:
        c = list_region[i]
        if c == '(':
            # find matching ')'
            p = i
            depth = 0
            while p < N:
                if list_region[p] == '(':
                    depth += 1
                elif list_region[p] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                p += 1
            if p >= N:
                # can't find matching ) - copy rest and break
                fixed_list_builder.append(list_region[i:])
                break
            tup = list_region[i:p+1]
            fixed_tup = fix_tuple_region(tup)
            fixed_list_builder.append(fixed_tup)
            i = p + 1
        else:
            fixed_list_builder.append(c)
            i += 1

    fixed_list_region = ''.join(fixed_list_builder)

    # Rebuild the full code
    fixed_code = code[:list_start] + fixed_list_region + code[list_end + 1:]
    return fixed_code


class LLMClient:
    """
    A unified client for interacting with Gemini, Ollama, or Azure OpenAI (ChatGPT).
    Usage:
        client = LLMClient(provider="gemini", ...)
        client = LLMClient(provider="ollama", ...)
        client = LLMClient(provider="azure_openai", ...)
        response = client.get_llm_response(prompt)
    """

    def __init__(
        self,
        provider: str = "gemini",  # "gemini", "ollama", or "azure_openai"
        gemini_model: str = "gemma-3-27b-it",
        ollama_model: str = "hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0",
        azure_openai_model: str = "gpt-4.1",
        azure_openai_endpoint: str = CHATGPT_ENDPOINT,
        azure_openai_api_key: str = CHATGPT_API_KEY,
        azure_openai_api_version: str = "2025-01-01-preview"
    ):
        self.provider = provider.lower()
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.azure_openai_model = azure_openai_model

        # Gemini
        self.gemini_client = None
        if self.provider == "gemini":
            if GEMINI_API_KEY:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                raise ValueError("Gemini API key must be provided.")

        # Ollama
        self.ollama_host = "http://[::1]:11434"
        self._ollama_client_instance = None
        if self.provider == "ollama":
            self._ollama_client_instance = ollama.Client(host=self.ollama_host)

        # Azure OpenAI
        self.azure_openai_client = None
        if self.provider == "azure_openai":
            self.azure_openai_client = AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                api_version=azure_openai_api_version,
            )

    def _get_response_from_gemini(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized.")
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model, contents=prompt)
        return response.text

    def _get_response_from_ollama(self, prompt: str) -> str:
        if not self._ollama_client_instance:
            raise RuntimeError("Ollama client not initialized.")
        response = self._ollama_client_instance.chat(
            model=self.ollama_model, messages=[
                {"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def _get_response_from_azure_openai(self, prompt: str) -> str:
        if not self.azure_openai_client:
            raise RuntimeError("Azure OpenAI client not initialized.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        completion = self.azure_openai_client.chat.completions.create(
            model=self.azure_openai_model,
            messages=messages,
            max_tokens=5000,
            temperature=1.0,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        # Parse the response to JSON, then extract the content
        response = json.loads(completion.to_json())[
            'choices'][0]['message']['content']
        return response

    def get_llm_response(self, prompt: str) -> str:
        """
        Gets a response from the configured LLM provider.
        Args:
            prompt: Either a string (for Gemini/Ollama) or a list of messages (for Azure OpenAI).
        Returns:
            The response text from the LLM.
        """
        if self.provider == "gemini":
            return self._get_response_from_gemini(prompt)
        elif self.provider == "ollama":
            return self._get_response_from_ollama(prompt)
        elif self.provider == "azure_openai":
            return self._get_response_from_azure_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def get_prompts_from_llm(
    prompt: str,
    llm_client: LLMClient,  # Accept the unified LLMClient instance
    parse_func: Callable = extract_and_parse_prompt_list,
    max_retries: int = 10
) -> List[PromptSet]:
    """
    Retrieves and parses a list of prompt-response pairs from an LLM.

    Args:
        prompt: The initial prompt to send to the LLM to generate the pairs.
        llm_client: An instance of LLMClient configured for Gemini or Ollama.
        parse_func: A function to parse the raw LLM response into a list of tuples.
                    Defaults to extract_and_parse_prompt_list.
        max_retries: The maximum number of attempts to get and parse a valid response.

    Returns:
        A list of (prompt_string, response_string) tuples.

    Raises:
        RuntimeError: If unable to get and parse prompts after multiple attempts.
        ValueError: If the LLM response does not contain a valid Python block
                    or if parsing fails.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Use the unified LLMClient to get the raw response
            raw = llm_client.get_llm_response(prompt)
            # print(f"Raw response on attempt {attempt}: {raw}...")

            # 1) extract the python block

            m = re.search(r'```python\s*([\s\S]*?)\s*```', raw)
            if not m:
                raise ValueError("No ```python ... ``` block found")
            code_block = m.group(1)

            # 2) normalize all literals to double-quoted form
            code_block = _fix_quote_issues(code_block)

            # print(f"Normalized code on attempt {attempt}: {code}...")

            # 3) convert the string to a list of tuples
            prompts_list = parse_func(code_block)
            prompts: List[Tuple[str, ...]] = prompts_list
            print(f"Loaded {len(prompts)} prompt-pairs.")
            print("First prompt:", prompts[0])
            return prompts

        except Exception as e:
            print(
                f"[Warning] get_prompts_from_llm parse error on attempt {attempt}/{max_retries}: {e}")
            print("raw response was:", raw)
            # sleep for 2 secs per attempt
            time.sleep(2 * attempt)  # exponential backoff

            if attempt == max_retries:
                raise RuntimeError(
                    "Failed to parse prompts after multiple attempts") from e
            # otherwise, retry immediately

    # Should never reach here
    raise RuntimeError("Unreachable")


class PriorityQueue:
    # type: ignore
    def __init__(self, max_capacity: int = 10, initial: Optional[List[Tuple[PromptSet, float]]] = None, filter_threshold: float = 0.01):

        self.filter_threshold = filter_threshold
        self.max_capacity: int = max_capacity
        # Store (score, prompt_set); min-heap root is the lowest score.
        self._heap: List[Tuple[float, PromptSet]] = []  # type: ignore

        # Track prompt sets for O(1) membership checks
        self._set: Set[PromptSet] = set()

        # If the user passed some initial prompt-sets, insert them now:
        if initial is not None:
            for prompt_set, score in initial:
                self.insert(prompt_set, score)

    # type: ignore
    def insert(self, prompt_set: PromptSet, score: float) -> None:
        # Skip if prompt set already exists
        if prompt_set in self._set:
            return
        # Skip low scores
        if score < self.filter_threshold:
            return

        if len(self._heap) < self.max_capacity:
            # Add new entry
            heapq.heappush(self._heap, (score, prompt_set))
            self._set.add(prompt_set)
        else:
            # Only replace if new score beats the current minimum
            if score > self._heap[0][0]:
                # Replace smallest entry, capturing the popped item
                old_score, old_set = heapq.heapreplace(
                    self._heap, (score, prompt_set))
                # Update prompt set
                self._set.remove(old_set)
                self._set.add(prompt_set)

    def get_best(self) -> Optional[Tuple[PromptSet, float]]:
        if not self._heap:
            return None
        best_score, best_set = max(self._heap, key=lambda x: x[0])
        return best_set, best_score

    def get_best_n(self, n: int, isNormalizedInts: bool = False) -> List[Tuple[PromptSet, float]]:
        if n <= 0:
            return []
        top_n = sorted(self._heap, key=lambda x: x[0], reverse=True)[:n]

        if isNormalizedInts:
            # Normalize scores to [60, 90] range
            min_score = 60
            max_score = 90

            raw_scores = [score for score, _ in top_n]
            mn, mx = min(raw_scores), max(raw_scores)

            if mx == mn:
                # everyone identical → give all max_score
                norm_scores = [max_score] * len(raw_scores)
            else:
                norm_scores = [
                    int(round((s - mn) / (mx - min_score) *
                        (max_score - min_score) + min_score))
                    for s in raw_scores
                ]

            return [(set_, norm) for (_, set_), norm in zip(top_n, norm_scores)]

        return [(set_, score) for score, set_ in top_n]

    def __len__(self) -> int:
        return len(self._heap)

    def __str__(self) -> str:
        ordered = sorted(self._heap, key=lambda x: x[0], reverse=True)
        return str([(set_, score) for score, set_ in ordered])

    def get_roulette_wheel_selection(self, n: int, isNormalizedInts: bool = True) -> List[Tuple[PromptSet, float]]:
        """
        Perform roulette-wheel (fitness-proportional) selection without replacement.

        Args:
            n: number of items to select.

        Returns:
            A list of up to n (prompt_set, score) tuples,
            selected without replacement according to fitness weights.
        """
        # Work on a temporary copy of the heap data
        pool = list(self._heap)  # each element is (score, prompt_set)
        total_fitness = sum(score for score, _ in pool)
        selected: List[Tuple[PromptSet, float]] = []

        # Don't request more than available
        n = min(n, len(pool))

        for _ in range(n):
            # Normalize weights and pick one
            r = random.random() * total_fitness
            cum = 0.0
            for idx, (score, set_) in enumerate(pool):
                cum += score
                if cum >= r:
                    # select this individual
                    selected.append((set_, score))
                    # remove it from pool & update total fitness
                    total_fitness -= score
                    pool.pop(idx)
                    break

        if not selected:
            return []

        if not isNormalizedInts:
            # If we want raw scores, return them directly
            return selected

        # --- Step B: normalize the selected raw scores to [min_score..max_score] ints ---
        min_score = 60
        max_score = 90

        raw_scores = [score for (_, score) in selected]
        mn, mx = min(raw_scores), max(raw_scores)

        if mx == mn:
            # everyone identical → give all max_score
            norm_scores = [max_score] * len(raw_scores)
        else:
            norm_scores = [
                int(round((s - mn) / (mx - mn) * (max_score - min_score) + min_score))
                for s in raw_scores
            ]

        # --- Step C: assemble final list ---
        selected_normalized: List[Tuple[PromptSet, int]] = [
            (set_, norm)
            for (set_, _), norm in zip(selected, norm_scores)
        ]
        return selected_normalized

    def get_average_score(self, top_n) -> float:
        """
        Calculate the average score of the top n items in the queue.
        If there are fewer than n items, it averages all available.
        """
        if not self._heap:
            return 0.0
        top_items = self.get_best_n(top_n)
        if not top_items:
            return 0.0
        total_score = sum(score for _, score in top_items)
        return total_score / len(top_items)

    def delete_top_n(self, n: int) -> None:
        """
        Remove the top n items from the priority queue.
        If n exceeds the current size, it removes all items.
        """
        if n <= 0:
            return
        for _ in range(min(n, len(self._heap))):
            if self._heap:
                score, set_ = heapq.heappop(self._heap)
                self._set.remove(set_)


def load_initial_prompts(path: str) -> List[InitialItem]:
    """
    Reads a text file where each line is of the form:
    ('feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'), Score: 0.9364
    and returns a list of ((tuple of 5 strings), score) tuples.
    """
    results = []

    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                # Fix quote issues in the line first
                fixed_line = _fix_quote_issues(f"prompts = [{line}]")

                # Extract the fixed content back (remove the wrapper)
                m = re.search(r'\[(.*)\]', fixed_line)
                if m:
                    fixed_line = m.group(1)

                # Now try to parse the tuple and score
                # Split at the last comma to separate tuple from score
                parts = fixed_line.rsplit(',', 1)
                if len(parts) != 2:
                    print(f"Warning: Could not split line {line_num}: {line}")
                    continue

                tuple_part = parts[0].strip()
                score_part = parts[1].strip()

                # Parse the tuple
                prompts = ast.literal_eval(tuple_part)
                if not isinstance(prompts, tuple) or len(prompts) != 5:
                    print(
                        f"Warning: Invalid tuple format at line {line_num}: {line}")
                    continue

                # Parse the score (remove "Score:" prefix)
                score_match = re.search(r'Score:\s*([\d.]+)', score_part)
                if not score_match:
                    print(
                        f"Warning: Could not parse score at line {line_num}: {line}")
                    continue

                score = float(score_match.group(1))
                results.append((prompts, score))

            except Exception as e:
                print(f"Error parsing line {line_num}: {line}")
                print(f"Error details: {e}")
                continue

    return results


def select_balanced_few_shot_subset(features: torch.Tensor, labels: torch.Tensor, n_per_class: int = 8, seed: int = 42):
    """
    Selects a balanced few-shot subset from the dataset.
    Returns features and labels for n_per_class samples per class.

    Args:
        features (torch.Tensor): Feature tensor of shape (N, D)
        labels (torch.Tensor): Label tensor of shape (N,)
        n_per_class (int): Number of samples to select per class
        seed (int): Random seed for reproducibility

    Returns:
        (torch.Tensor, torch.Tensor): Subset features and labels
    """
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Find unique classes
    classes = torch.unique(labels)
    selected_indices = []

    for cls in classes:
        # Get indices for this class
        cls_indices = (labels == cls).nonzero(as_tuple=True)[0]
        # Shuffle indices with the seeded generator
        cls_indices = cls_indices[torch.randperm(
            len(cls_indices), generator=generator)]
        # Select up to n_per_class samples
        selected_indices.extend(cls_indices[:n_per_class].tolist())

    # Shuffle all selected indices to mix classes with the seeded generator
    selected_indices = torch.tensor(selected_indices)
    selected_indices = selected_indices[torch.randperm(
        len(selected_indices), generator=generator)]

    # Subset features and labels
    subset_features = features[selected_indices]
    subset_labels = labels[selected_indices]

    return subset_features, subset_labels


def load_last_iteration_prompts(path: str) -> List[InitialItem]:
    """
    Reads the last iteration's prompt progression file, which has the format:
    iteration 1: 
    ('neg', 'pos'), Score: 0.9364
    ('neg2', 'pos2'), Score: 0.8456
    ....

    iteration 'n': 
    ('neg3', 'pos3'), Score: 0.9123
    ('neg4', 'pos4'), Score: 0.7890

    Returns a list of ((neg, pos), score) tuples from the last iteration.
    """
    last_iteration_prompts = []
    current_iteration = None

    with open(path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                # Check if this line indicates a new iteration
                iteration_match = re.match(r'[Ii]teration (\d+):\s*$', line)
                if iteration_match:
                    # If we found a previous iteration, store its prompts
                    if current_iteration is not None:
                        last_iteration_prompts = []  # Reset for new iteration
                    current_iteration = int(iteration_match.group(1))
                    continue

                # Try to parse prompt lines: ('neg', 'pos'), Score: 0.9364
                pattern = r"\('([^']*)', '([^']*)'\), Score: ([\d.]+)"
                match = re.match(pattern, line)

                if match and current_iteration is not None:
                    neg_prompt = match.group(1)
                    pos_prompt = match.group(2)
                    score = float(match.group(3))

                    prompt_pair = (neg_prompt, pos_prompt)
                    last_iteration_prompts.append((prompt_pair, score))
                elif not iteration_match:  # Don't warn about iteration headers
                    print(f"Warning: Could not parse line {line_num}: {line}")

            except Exception as e:
                print(f"Error parsing line {line_num}: {line}")
                print(f"Error details: {e}")
                continue

    # Return the prompts from the last iteration found
    return last_iteration_prompts


# def load_last_iteration_prompts(path: str) -> List[InitialItem]:
#     """
#     Reads the last iteration's prompt progression file, which has the format:
#     iteration 1:
#     ('prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5'), Score: 0.9364
#     ('prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5'), Score: 0.8456
#     ....

#     iteration 'n':
#     ('prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5'), Score: 0.9123
#     ('prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5'), Score: 0.7890

#     Returns a list of ((tuple of 5 strings), score) tuples from the last iteration.
#     """
#     last_iteration_prompts = []
#     current_iteration = None

#     with open(path, 'r', encoding='utf-8') as file:
#         for line_num, line in enumerate(file, 1):
#             line = line.strip()
#             if not line:  # Skip empty lines
#                 continue

#             try:
#                 # Check if this line indicates a new iteration
#                 iteration_match = re.match(r'[Ii]teration (\d+):\s*$', line)
#                 if iteration_match:
#                     # If we found a previous iteration, store its prompts
#                     if current_iteration is not None:
#                         last_iteration_prompts = []  # Reset for new iteration
#                     current_iteration = int(iteration_match.group(1))
#                     continue

#                 # Skip iteration headers
#                 if current_iteration is None:
#                     continue

#                 # Fix quote issues in the line first
#                 fixed_line = _fix_quote_issues(f"prompts = [{line}]")

#                 # Extract the fixed content back (remove the wrapper)
#                 m = re.search(r'\[(.*)\]', fixed_line)
#                 if m:
#                     fixed_line = m.group(1)

#                 # Split at the last comma to separate tuple from score
#                 parts = fixed_line.rsplit(',', 1)
#                 if len(parts) != 2:
#                     print(f"Warning: Could not split line {line_num}: {line}")
#                     continue

#                 tuple_part = parts[0].strip()
#                 score_part = parts[1].strip()

#                 # Parse the tuple
#                 prompts = ast.literal_eval(tuple_part)
#                 if not isinstance(prompts, tuple) or len(prompts) != 5:
#                     print(
#                         f"Warning: Invalid tuple format at line {line_num}: {line}")
#                     continue

#                 # Parse the score (remove "Score:" prefix)
#                 score_match = re.search(r'Score:\s*([\d.]+)', score_part)
#                 if not score_match:
#                     print(
#                         f"Warning: Could not parse score at line {line_num}: {line}")
#                     continue

#                 score = float(score_match.group(1))
#                 last_iteration_prompts.append((prompts, score))

#             except Exception as e:
#                 print(f"Error parsing line {line_num}: {line}")
#                 print(f"Error details: {e}")
#                 continue

#     # Return the prompts from the last iteration found
#     return last_iteration_prompts
