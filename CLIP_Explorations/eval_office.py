import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import requests
import itertools
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, Compose, Resize, ToTensor
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from datasets import load_dataset
import numpy as np


# --- Helper: build a DataLoader for a single domain ---
def build_pacs_loader(processor, domain, split='train', batch_size=128, num_workers=4):
    # 1) Load HF PACS and filter to this domain
    ds = load_dataset("flwrlabs/office-home", split=split)
    # print unique domains
    # print(ds.unique('domain'))
    ds = ds.filter(lambda ex: ex['domain'] == domain)

    # 2) Define a transform + collate_fn
    image_transform = Compose([
        Resize((224, 224)),
    ])

    def collate_fn(batch):
        # apply the PIL transform, gather labels
        images = [image_transform(ex['image']).convert("RGB") for ex in batch]
        labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.long)
        # tokenize / preprocess images in one go
        inputs = processor(images=images, return_tensors="pt", padding=True)
        return inputs, labels

    # 3) Wrap in DataLoader
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )


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
    prompts = [f"a photo of {c}" for c in classes]
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
    prompts = [f"a photo of {c}" for c in classes]

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
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    adapter = Adapter(dim=512).to(device)
    adapter.load_state_dict(torch.load(
        "adapter.pth", map_location=device))
    adapter.eval()

    print("Precomputing Image Features...")
    all_feats = []
    all_labels = []

    # office-home classes
    classes_str = "Alarm Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk Lamp, Drill, Eraser, Exit Sign, Fan,File Cabinet, Flipflops, Flowers, Folder, Fork, Glasses, Hammer, Helmet, Kettle, Keyboard,Knives, Lamp Shade, Laptop, Marker, Monitor, Mop, Mouse, Mug, Notebook, Oven, Pan,Paper Clip, Pen, Pencil, Postit Notes, Printer, Push Pin, Radio, Refrigerator, ruler,Scissors, Screwdriver, Shelf, Sink, Sneakers, Soda, Speaker, Spoon, Table, Telephone,Toothbrush, Toys, Trash Can, TV, Webcam"
    classes = classes_str.split(",")
    classes = [c.lower().strip() for c in classes]
    print(f"Number of classes: {len(classes)}")

    # Domains in Office-Home
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    for domain in domains:
        print(f"\n=== Processing domain: {domain} ===")
        loader = build_pacs_loader(
            processor, domain, split='train', batch_size=128)

        # 1) Precompute all image features & labels in this domain
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Extracting feats [{domain}]"):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
                all_labels.append(labels)

        image_feats = torch.cat(all_feats, dim=0)
        image_labels = torch.cat(all_labels, dim=0)

        # 2) Zero-shot CLIP (no adapter)
        results_no_adapter = evaluate_clip_prompts(
            image_feats=image_feats,
            image_labels=image_labels,
            model=model,
            processor=processor,
            classes=classes
        )
        print(f"[{domain}] Zero-Shot Accuracy: {results_no_adapter['accuracy']:.4f}  AUC: {results_no_adapter['auc']:.4f}")

        # 3) Zero-shot CLIP + Adapter
        results_with_adapter = evaluate_clip_with_adapter(
            image_feats=image_feats,
            image_labels=image_labels,
            model=model,
            processor=processor,
            adapter=adapter,
            classes=classes
        )
        print(f"[{domain}] Adapter Accuracy:  {results_with_adapter['accuracy']:.4f}  AUC: {results_with_adapter['auc']:.4f}")


if __name__ == "__main__":
    main()
