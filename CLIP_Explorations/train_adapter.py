import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from PIL import Image
import requests
import itertools
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import optim

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


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

# ------------------ Feature Triplet Dataset ------------------


class FeatureTripletDataset(Dataset):
    """
    Constructs triplets (anchor, positive, negative) from text_features.
    text_features: dict[class_key -> Tensor(num_domains, dim)] or list of Tensors.
    """

    def __init__(self, text_features):
        if isinstance(text_features, dict):
            self.class_keys = list(text_features.keys())
            self.features = [text_features[k] for k in self.class_keys]
        else:
            self.features = text_features
            self.class_keys = list(range(len(self.features)))
        # convert all features to CPU float tensors
        self.features = [feat.detach().cpu().float() for feat in self.features]
        self.lengths = [feat.size(0) for feat in self.features]
        self.num_classes = len(self.features)
        if self.num_classes < 2:
            raise ValueError("Need at least 2 classes for triplet generation")

    def __len__(self):
        # define epoch length as total possible anchors
        return sum(self.lengths)

    def __getitem__(self, idx):  # idx ignored for random sampling
        # Sample a class with at least 2 samples
        cls = random.choice([c for c, l in enumerate(self.lengths) if l >= 2])
        n_samples = self.lengths[cls]
        a_idx, p_idx = random.sample(range(n_samples), 2)
        anchor = self.features[cls][a_idx]
        positive = self.features[cls][p_idx]

        # Sample negative class and sample
        neg_cls = random.choice(
            [c for c in range(self.num_classes) if c != cls])
        neg_idx = random.randrange(self.lengths[neg_cls])
        negative = self.features[neg_cls][neg_idx]

        return anchor, positive, negative

# ------------------ Triplet Training ------------------


def cosine_dist(x, y):
    # returns 1 − cosine_similarity, so smaller means more similar
    return 1.0 - F.cosine_similarity(x, y, dim=1)


def train_adapter(
    text_features, device='cuda',
    lr=1e-3, batch_size=256, epochs=10,
    num_workers=0
):
    """
    Train an Adapter using triplet loss on precomputed text_features.

    Args:
        text_features: dict or list of Tensor(num_domains, dim)
        device: 'cuda' or 'cpu'
        lr: learning rate
        batch_size: batch size
        epochs: number of epochs
        num_workers: DataLoader worker count (0 for debug)

    Returns:
        Trained Adapter model
    """
    dataset = FeatureTripletDataset(text_features)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )

    # infer feature dimension
    sample_feat = next(iter(dataset))[0]
    dim = sample_feat.size(0)

    adapter = Adapter(dim).to(device)
    loss_fn = nn.TripletMarginWithDistanceLoss(
        margin=0.2, distance_function=cosine_dist, reduction='mean')
    optimizer = optim.Adam(adapter.parameters(), lr=lr)

    adapter.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch}"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            out_a = adapter(anchor)
            out_p = adapter(positive)
            out_n = adapter(negative)

            loss = loss_fn(out_a, out_p, out_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

    return adapter


def main():

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    classes = [
        'a dog',
        'an elephant',
        'a giraffe',
        'a guitar',
        'a horse',
        'a house',
        'a person',
    ]

    domains = [
        'a photo',
        'a painting',
        'a sculpture',
        'a drawing',
        'a cartoon',
        'a sketch',
        'a diagram',
        'a model',
        'a figure',
        'a photograph',
        'a mural',
        'a fresco',
        'a lithograph',
        'an etching',
        'a print',
        'a poster',
        'a mixed-media piece',
        'a textile',
        'an animation',
        'a 3D model',
        'a collage',
        'a tapestry',
        'an illustration',
        'a caricature',
    ]

    # Components to generate new domain phrases
    materials = ['watercolor', 'oil', 'charcoal', 'ink',
                 'pastel', 'digital', 'acrylic', 'graphite', 'chalk']
    styles = ['abstract', 'realistic', 'minimalist', 'expressionist',
              'impressionist', 'surrealist', 'cubist', 'geometric', 'pop-art']
    formats = ['panorama', 'portrait', 'landscape', 'macro',
               'micro', 'aerial', 'fisheye', 'wide-angle', 'time-lapse']
    subtypes = ['illustration', 'rendering', 'blueprint',
                'schematic', 'map', 'chart', 'poster', 'collage', 'mosaic']

    # --- Start of New Domain Expansion Logic ---

    # Store already added domains to avoid duplicates if categories overlap
    # Initialize with existing domains to avoid adding them again if generated
    existing_phrases = set(domains)

    # Target number of domains (can be adjusted)
    TARGET_DOMAIN_COUNT = 20000  # You had 1009, we can aim for something similar or more

    # Combination 1: Style + Material + Subtype (as you had)
    for style, material, subtype_item in itertools.product(styles, materials, subtypes):
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            break
        phrase = f"a {style} {material} {subtype_item}"
        if phrase not in existing_phrases:
            domains.append(phrase)
            existing_phrases.add(phrase)
    if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
        print(f"Reached target after Style + Material + Subtype combinations.")

    # Combination 2: Style + Material + Format
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for style, material, format_item in itertools.product(styles, materials, formats):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            phrase = f"a {style} {material} {format_item}"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Style + Material + Format combinations.")

    # Combination 3: Style + Subtype + Format
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for style, subtype_item, format_item in itertools.product(styles, subtypes, formats):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            # slightly different phrasing for variety
            phrase = f"a {style} {subtype_item} in {format_item} format"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Style + Subtype + Format combinations.")

    # Combination 4: Material + Subtype + Format
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for material, subtype_item, format_item in itertools.product(materials, subtypes, formats):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            # alternative phrasing
            phrase = f"a {material} {subtype_item} ({format_item})"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Material + Subtype + Format combinations.")

    # Combination 5: Style + Format
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for style, format_item in itertools.product(styles, formats):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            phrase = f"a {style} {format_item}"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Style + Format combinations.")

    # Combination 6: Material + Subtype
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for material, subtype_item in itertools.product(materials, subtypes):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            phrase = f"a {material} {subtype_item}"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Material + Subtype combinations.")

    # Combination 7: Style + Material (often a painting or drawing implicitly)
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        for style, material in itertools.product(styles, materials):
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            phrase = f"a {style} {material} piece"
            if phrase not in existing_phrases:
                domains.append(phrase)
                existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(f"Reached target after Style + Material combinations.")

    # Combination 8: Adding "detailed" or "simple" prefix to some combinations
    adjectives = ["detailed", "simple", "complex", "minimal"]
    if len(existing_phrases) < TARGET_DOMAIN_COUNT:
        # Example: Adjective + Style + Material + Subtype
        # Taking a slice of products to avoid excessive computation if target is close
        temp_products = list(itertools.product(styles, materials, subtypes))[:(
            TARGET_DOMAIN_COUNT - len(existing_phrases)) * 2 // len(adjectives) + 1]  # Heuristic limit
        for adj in adjectives:
            if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                break
            for style, material, subtype_item in temp_products:
                if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
                    break
                phrase = f"a {adj} {style} {material} {subtype_item}"
                if phrase not in existing_phrases:
                    domains.append(phrase)
                    existing_phrases.add(phrase)
        if len(existing_phrases) >= TARGET_DOMAIN_COUNT:
            print(
                f"Reached target after Adjective + Style + Material + Subtype combinations.")

    print(f"Number of domains: {len(domains)}")
    print(f"Domains: {domains}")

    # Initialize storage for text features
    text_features = {cls: None for cls in classes}

    # Loop over each class and process all domains in one batch
    for cls in classes:
        # Build all text prompts for this class
        texts = [f"{domain} of {cls}" for domain in domains]

        # Tokenize the entire batch at once
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Extract features in one forward pass
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            # Normalize embeddings
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Store tensor of shape (num_domains, feature_dim)
        text_features[cls] = feats.cpu()

    # Example access:
    # text_features['a dog'] is a tensor of size (len(domains), feature_dim)

    print({cls: feats.shape for cls, feats in text_features.items()})
    adapter = train_adapter(text_features, device='cuda')
    torch.save(adapter.state_dict(), "adapter.pth")

    # visualization
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([tf.numpy()
            for (key, tf) in text_features.items()], axis=0))

    # plot PCA on same figure
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    colors = sns.color_palette("deep", len(classes))
    i = 0
    for cls, tf in text_features.items():
        pca_result = pca.transform(tf.numpy())
        plt.scatter(pca_result[:, 0], pca_result[:, 1],
                    label=cls, color=colors[i])
        i += 1

    plt.title("PCA of Text Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig("pca_text_features.png")

    # Apply adapter to text features
    adapted_features = {cls: None for cls in classes}
    for cls in classes:
        # Apply adapter to each class
        with torch.no_grad():
            adapted_features[cls] = adapter(
                text_features[cls].to(device)).cpu()

    pca = PCA(n_components=2)

    pca.fit(np.concatenate([tf.numpy()
            for (key, tf) in adapted_features.items()], axis=0))

    # plot PCA on same figure
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    colors = sns.color_palette("deep", len(classes))
    i = 0
    for cls, tf in adapted_features.items():
        pca_result = pca.transform(tf.numpy())
        plt.scatter(pca_result[:, 0], pca_result[:, 1],
                    label=cls, color=colors[i])
        i += 1

    plt.title("PCA of Text Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()
    plt.savefig("pca_text_features_adapter.png")
    plt.close()


if __name__ == "__main__":
    main()
