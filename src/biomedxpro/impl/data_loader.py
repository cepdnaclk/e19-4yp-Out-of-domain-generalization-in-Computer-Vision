# src/biomedxpro/impl/data_loader.py
"""
BiomedDataLoader: The Infrastructure Service for Data Encoding and Caching

This module provides the main data processing pipeline. It:
1. Accepts a list of StandardSample objects from an adapter
2. Loads the BioMedCLIP model (expensive operation, done once)
3. Batches and encodes images into embeddings
4. Caches the encoded dataset to disk for future runs
5. Returns an EncodedDataset for the evolutionary engine

The loader is dataset-agnostic: it doesn't care if samples came from
Derm7pt, Camelyon17, or any other dataset. It just processes whatever
the adapter provides.

Design Principles:
- Lazy Loading: Model is loaded only when needed.
- Caching: Disk cache prevents expensive re-encoding.
- Batch Processing: GPU is utilized efficiently via PyTorch DataLoader.
- Memory Management: Processed data lives on the specified device (GPU/CPU).
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from biomedxpro.core.domain import EncodedDataset, StandardSample

logger = logging.getLogger(__name__)


class BiomedCLIPModel:
    """
    Lazy-loading wrapper for the BioMedCLIP model.

    The model is expensive to load (600MB+), so we defer loading until
    the first encode() call. This allows the system to run without a
    loaded model if only using cached data.
    """

    _instance: Optional["BiomedCLIPModel"] = None
    _model = None
    _preprocess = None
    _tokenizer = None

    def __new__(cls) -> "BiomedCLIPModel":
        """Singleton pattern: ensure only one model instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def ensure_loaded(self, device: torch.device) -> None:
        """Load the model and processor if not already loaded."""
        if self._model is not None:
            return  # Already loaded

        logger.info("Loading BioMedCLIP model using open_clip...")
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer

            model_name = (
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self._model, self._preprocess = create_model_from_pretrained(model_name)
            self._tokenizer = get_tokenizer(model_name)

            self._model.to(device)
            self._model.eval()

            logger.info(f"BioMedCLIP loaded on {device}")
        except ImportError:
            raise ImportError(
                "Required libraries not installed. "
                "Please install: pip install open_clip_torch transformers"
            )

    def encode_images(
        self,
        image_paths: list[str],
        device: torch.device,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """
        Encode a list of images into embeddings.

        Args:
            image_paths: List of paths to image files.
            device: Device to process on (cuda or cpu).
            batch_size: Number of images per batch.
            num_workers: Number of workers for data loading.

        Returns:
            Tensor of shape (len(image_paths), embedding_dim) where embedding_dim=512.
        """
        self.ensure_loaded(device)
        assert self._model is not None
        assert self._preprocess is not None

        # Create a simple dataset and dataloader for batch processing
        # We pass the preprocess function from open_clip
        image_dataset = ImagePathDataset(image_paths, transform=self._preprocess)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        all_embeddings = []
        is_cuda = device.type == "cuda"
        first_batch = True
        with torch.no_grad():
            for batch_images, _ in tqdm.tqdm(dataloader, desc="Encoding images"):
                batch_images = batch_images.to(device, non_blocking=True)

                if first_batch:
                    if not is_cuda:
                        logger.warning(
                            f"⚠️ PROCESSING ON CPU! Images are on {batch_images.device}"
                        )
                    else:
                        logger.info(
                            f"✅ Processing on GPU: {torch.cuda.get_device_name(0)}"
                        )
                    first_batch = False

                # Mixed Precision provides a significant speedup on modern GPUs
                with torch.amp.autocast("cuda", enabled=is_cuda):
                    features = self._model.encode_image(batch_images)
                    # Normalize embeddings
                    features = features / features.norm(dim=-1, keepdim=True)

                all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings, dim=0).to(device)


class ImagePathDataset(Dataset):
    """
    Simple dataset that loads and transforms images from file paths.
    """

    def __init__(self, image_paths: list[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, path
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            # Return dummy tensor to keep the batch size consistent
            return torch.zeros((3, 224, 224)), path


class BiomedDataLoader:
    """
    The main data processing service.

    Responsibilities:
    1. Accept a list of StandardSample objects.
    2. Check if an encoded cache exists on disk.
    3. If cache hit: Load the cached tensors.
    4. If cache miss: Encode images using BioMedCLIP and save to cache.
    5. Return an EncodedDataset for the engine.
    """

    def __init__(
        self,
        cache_dir: str = ".biomedxpro_cache",
        batch_size: int = 32,
        device: Optional[str] = None,
        num_workers: int = 4,
    ):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to store encoded dataset caches.
            batch_size: Batch size for encoding.
            device: Device to process on. If None, auto-detect (cuda if available, else cpu).
            num_workers: Number of workers for data loading.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            f"BiomedDataLoader initialized. Device: {self.device}, Workers: {self.num_workers}"
        )

    def load_encoded_dataset(
        self,
        name: str,
        samples: list[StandardSample],
        class_names: list[str],
    ) -> EncodedDataset:
        """
        Load or create an encoded dataset.

        If a cached version exists, load it. Otherwise, encode the samples,
        save to cache, and return the encoded dataset.

        Args:
            name: Name for this dataset (used for caching and logs).
            samples: List of StandardSample objects from an adapter.
            class_names: Names of the classes (e.g., ["Benign", "Malignant"]).

        Returns:
            An EncodedDataset ready for the evolutionary engine.
        """
        # Generate a cache key based on the samples
        cache_key = self._compute_cache_key(samples)
        cache_path = self.cache_dir / f"{name}_{cache_key}.pt"

        # Try to load from cache
        if cache_path.exists():
            logger.info(f"Loading encoded dataset from cache: {cache_path}")
            return self._load_from_cache(cache_path, name, class_names)

        # Cache miss: encode the samples
        logger.info(f"Cache miss for {name}. Encoding {len(samples)} samples...")

        # Extract image paths and labels
        image_paths = [sample.image_path for sample in samples]
        labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)

        # Encode images using BioMedCLIP
        features = BiomedCLIPModel().encode_images(
            image_paths,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        # Create the encoded dataset
        encoded_dataset = EncodedDataset(
            name=name,
            features=features,
            labels=labels.to(self.device),
            class_names=class_names,
        )

        # Save to cache
        logger.info(f"Caching encoded dataset to: {cache_path}")
        self._save_to_cache(encoded_dataset, cache_path)

        return encoded_dataset

    def _compute_cache_key(self, samples: list[StandardSample]) -> str:
        """
        Compute a hash of the samples to use as a cache key.

        This ensures that different sets of samples (or reordered samples)
        get different cache files.
        """
        content = "\n".join(f"{s.image_path}:{s.label}" for s in samples)
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]

    def _save_to_cache(self, dataset: EncodedDataset, cache_path: Path) -> None:
        """Save an encoded dataset to disk."""
        cache_data = {
            "name": dataset.name,
            "features": dataset.features.cpu(),  # Save on CPU to reduce file size
            "labels": dataset.labels.cpu(),
            "class_names": dataset.class_names,
        }
        torch.save(cache_data, cache_path)
        logger.info(f"Dataset cached: {cache_path}")

    def _load_from_cache(
        self, cache_path: Path, name: str, class_names: list[str]
    ) -> EncodedDataset:
        """Load an encoded dataset from disk."""
        cache_data = torch.load(cache_path, map_location=self.device)
        return EncodedDataset(
            name=name,
            features=cache_data["features"].to(self.device),
            labels=cache_data["labels"].to(self.device),
            class_names=class_names,
        )

    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache cleared: {self.cache_dir}")
