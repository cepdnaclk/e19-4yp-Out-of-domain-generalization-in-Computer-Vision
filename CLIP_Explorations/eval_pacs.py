import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import requests
import itertools
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from tqdm import tqdm
import deeplake

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_clip_prompts(
    image_feats: torch.Tensor,    # (N, D), precomputed
    image_labels: torch.Tensor,   # (N,), integer labels 0..C-1
    model,
    processor,
    classes: list            # list of class strings, e.g. ['a dog', ...]
):
    """
    Evaluates CLIP zero-shot using 'a photo of {class}' prompts for each class.
    Returns accuracy, auc per class vs rest, confusion matrix, and classification report.
    """
    device = next(model.parameters()).device
    # prepare text prompts
    prompts = [
        f"a photo of {cls.replace('a ', '').replace('an ', '')}" for cls in classes]
    text_inputs = processor(
        text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**text_inputs)  # (C, D)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        # image_feats may already be normalized; ensure floats on device
        feats = image_feats.to(device).float()              # (N, D)
        feats = feats / feats.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * feats @ text_feats.t()      # (N, C)
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

    y_true = image_labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # compute metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4)
    # For multi-class AUC, compute one-vs-rest
    try:
        auc = roc_auc_score(
            y_true, probs.cpu().numpy(), multi_class='ovr'
        )
    except Exception:
        auc = None
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}


def evaluate_clip_with_adapter(
    image_feats: torch.Tensor,
    image_labels: torch.Tensor,
    model,
    processor,
    adapter,
    classes: list
):
    """
    Same as evaluate_clip_prompts, but applies adapter to both text and image features.
    """
    device = next(model.parameters()).device
    prompts = [
        f"a photo of {cls.replace('a ', '').replace('an ', '')}" for cls in classes]
    text_inputs = processor(
        text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        # text side
        text_feats = model.get_text_features(**text_inputs)  # (C, D)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        text_feats = adapter(text_feats)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        # image side
        feats = image_feats.to(device).float()
        feats = feats / feats.norm(dim=1, keepdim=True)
        feats = adapter(feats)
        feats = feats / feats.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * feats @ text_feats.t()      # (N, C)
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

    y_true = image_labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4)
    try:
        auc = roc_auc_score(
            y_true, probs.cpu().numpy(), multi_class='ovr'
        )
    except Exception:
        auc = None
    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}


def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    adapter = torch.load("adapter.pth")
    adapter.to(device)
    adapter.eval()

    print("=== Loading PACS Dataset ===")
    ds = deeplake.load("hub://activeloop/pacs-test")

    print("=== Loading Images ===")
    loader = ds.pytorch(
        batch_size=128,
        shuffle=False,
        num_workers=0,
        # Specify decoding images as PIL objects
        decode_method={'images': 'pil'},
        # Keep transforms=None or only specify for other tensors if needed
        # Or simply remove if only images/labels/domains are loaded
        transforms={'labels': None, 'domains': None},
    )

    print("Precomputing Image Features...")
    all_feats = []
    all_labels = []
    classes = [
        'a dog',
        'an elephant',
        'a giraffe',
        'a guitar',
        'a horse',
        'a house',
        'a person'
    ]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Batches"):
            # use the correct keys
            images = batch["images"]      # PIL.Image list of length B
            labels = batch["labels"]      # torch.Tensor of shape (B,)

            # CLIP preprocessing + feature extraction
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)      # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

            # Stack into single tensors
            image_feats = torch.cat(all_feats,  dim=0)  # (N, D)
            image_labels = torch.cat(all_labels, dim=0)  # (N,)

            # 3) Evaluate CLIP zero-shot (no adapter)
            results_no_adapter = evaluate_clip_prompts(
                image_feats=image_feats,
                image_labels=image_labels,
                model=model,
                processor=processor,
                classes=classes
            )

    print("=== Zero-Shot CLIP (no adapter) ===")
    print(f"Accuracy: {results_no_adapter['accuracy']:.4f}")
    print(f"AUC (OvR): {results_no_adapter['auc']:.4f}")
    print("Confusion Matrix:\n", results_no_adapter['cm'])
    print("Classification Report:\n", results_no_adapter['report'])

    results_with_adapter = evaluate_clip_with_adapter(
        image_feats=image_feats,
        image_labels=image_labels,
        model=model,
        processor=processor,
        adapter=adapter,
        classes=classes
    )

    print("\n=== Zero-Shot CLIP + Adapter ===")
    print(f"Accuracy: {results_with_adapter['accuracy']:.4f}")
    print(f"AUC (OvR): {results_with_adapter['auc']:.4f}")
    print("Confusion Matrix:\n", results_with_adapter['cm'])
    print("Classification Report:\n", results_with_adapter['report'])
