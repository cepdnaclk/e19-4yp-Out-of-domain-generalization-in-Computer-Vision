# src/biomedxpro/impl/adapters.py
"""
Dataset Adapters: Strategy Pattern for Multiple Dataset Formats

This module provides a registry system and concrete adapter implementations
for different biomedical datasets. Each adapter knows how to parse its specific
dataset format and convert it to StandardSample objects that the DataLoader
can process uniformly.

The Registry Pattern allows configuration-driven dataset selection without
hardcoding adapter choices in the main application.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from biomedxpro.core.domain import DataSplit, StandardSample
from biomedxpro.core.interfaces import IDatasetAdapter

# --- Registry System ---

_ADAPTER_REGISTRY: dict[str, type[IDatasetAdapter]] = {}


def register_adapter(
    name: str,
) -> Callable[[type[IDatasetAdapter]], type[IDatasetAdapter]]:
    """
    Decorator to register a dataset adapter in the global registry.

    Usage:
        @register_adapter("derm7pt")
        class Derm7ptAdapter(IDatasetAdapter):
            ...

    Args:
        name: The unique string key for this adapter (e.g., "derm7pt").

    Returns:
        A decorator function that registers the class.
    """

    def decorator(cls: type[IDatasetAdapter]) -> type[IDatasetAdapter]:
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str, **kwargs: Any) -> IDatasetAdapter:
    """
    Factory function: Lookup an adapter by its registered name.

    Args:
        name: The key to look up (e.g., "derm7pt").
        **kwargs: Optional arguments to pass to the adapter constructor.

    Returns:
        An instance of the registered adapter class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY.keys())
        raise KeyError(f"Adapter '{name}' not found. Available adapters: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def list_available_adapters() -> list[str]:
    """Return a list of all registered adapter names."""
    return list(_ADAPTER_REGISTRY.keys())


# --- Concrete Adapters ---


@register_adapter("derm7pt")
class Derm7ptAdapter(IDatasetAdapter):
    """
    Adapter for the Derm7pt dataset.

    Derm7pt is a dermoscopy dataset organized as:
    - meta.csv: Contains diagnosis, image filenames, and metadata.
    - train_indexes.csv, valid_indexes.csv, test_indexes.csv: Define splits.
    - images/: Directory containing .jpg image files.

    The adapter reads meta.csv, filters by the split's indexes, and maps
    the "diagnosis" column to binary labels (melanoma vs. benign).

    Supports few-shot learning scenarios where only N samples per class are used.
    """

    def __init__(
        self,
        root: str,
        shots: int = 0,
    ) -> None:
        """
        Initialize the Derm7pt adapter.

        Args:
            root: Root directory of Derm7pt (must contain meta.csv and images/).
            shots: Number of samples per class (0 = full dataset).
        """
        self.root = Path(root)
        self.shots = shots

    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        """
        Load Derm7pt samples for the specified split.

        Args:
            split: The split to load (TRAIN, VAL, or TEST).

        Returns:
            List of StandardSample objects (image_path, label).
        """
        # Map DataSplit enum to filename
        split_map = {
            DataSplit.TRAIN: "train_indexes.csv",
            DataSplit.VAL: "valid_indexes.csv",
            DataSplit.TEST: "test_indexes.csv",
        }

        meta_path = self.root / "meta" / "meta.csv"
        split_path = self.root / "meta" / split_map[split]
        image_base = self.root / "images"

        # Validate required files exist
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.csv not found at {meta_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"{split_map[split]} not found at {split_path}")
        if not image_base.exists():
            raise FileNotFoundError(f"images directory not found at {image_base}")

        # Load metadata and split indexes
        meta_df = pd.read_csv(meta_path)
        split_df = pd.read_csv(split_path)
        indexes = split_df["indexes"].tolist()

        # Filter metadata by split indexes
        split_df_filtered = meta_df.iloc[indexes].reset_index(drop=True)

        samples = []

        # Logic: We only apply 'shots' to the TRAIN split.
        # Validation and Test should be full for accurate evaluation.
        if self.shots == 0 or split != DataSplit.TRAIN:
            for _, row in split_df_filtered.iterrows():
                # Binary classification: melanoma vs. non-melanoma
                diagnosis = row.get("diagnosis", "").lower()
                label = 1 if "melanoma" in diagnosis else 0

                # Get image filename and construct full path
                image_filename = row.get("derm")
                if not image_filename:
                    continue

                image_path = image_base / image_filename

                # Only include samples with existing image files
                if not image_path.exists():
                    continue

                samples.append(StandardSample(image_path=str(image_path), label=label))

            return samples

        # Few-Shot Logic: Stratified Random Sampling

        # Fix seed for reproducibility
        random.seed(42)

        candidates_by_class = defaultdict(list)

        for _, row in split_df_filtered.iterrows():
            diagnosis = row.get("diagnosis", "").lower()
            label = 1 if "melanoma" in diagnosis else 0

            image_filename = row.get("derm")
            if not image_filename:
                continue

            image_path = image_base / image_filename

            if image_path.exists():
                candidates_by_class[label].append(
                    StandardSample(image_path=str(image_path), label=label)
                )

        # Sample k from each class
        for label, candidates in candidates_by_class.items():
            count = min(self.shots, len(candidates))
            samples.extend(random.sample(candidates, count))

        return samples


@register_adapter("camelyon17")
class Camelyon17Adapter(IDatasetAdapter):
    """
    Adapter for the Camelyon17 dataset (Wilds version).

    Camelyon17 is a histopathology dataset with:
    - metadata.csv: Contains tumor labels, center info, patient/node coordinates.
    - patient_XXX_node_Y/: Directories containing patch images.
    - Multiple centers (0-4) representing different medical institutions.

    The adapter reads metadata.csv, filters by center and split,
    constructs image paths, and creates binary labels (tumor vs. non-tumor).

    Supports few-shot learning scenarios where only N samples per class are used.
    """

    def __init__(
        self,
        root: str,
        shots: int = 0,
        train_centers: list[int] = [0, 1, 2],
        val_centers: list[int] = [3],
        test_centers: list[int] = [4],
    ) -> None:
        """
        Initialize the Camelyon17 adapter.

        Args:
            root: Root directory of Camelyon17 (must contain metadata.csv).
            shots: Number of samples per class (0 = full dataset).
            train_centers: List of center IDs to use for training.
            val_centers: List of center IDs to use for validation.
            test_centers: List of center IDs to use for testing.
        """
        self.root = Path(root)
        self.shots = shots
        self.train_centers = train_centers
        self.val_centers = val_centers
        self.test_centers = test_centers

    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        """
        Load Camelyon17 samples for the specified split.

        Args:
            split: The split to load (TRAIN, VAL, or TEST).

        Returns:
            List of StandardSample objects (image_path, label).
        """
        metadata_path = self.root / "metadata.csv"

        # Load metadata
        metadata_df = pd.read_csv(metadata_path)

        # Filter by split and center
        if split == DataSplit.TRAIN:
            # Training: use split=0 from specified centers
            filtered_df = metadata_df[
                (metadata_df["center"].isin(self.train_centers))
                & (metadata_df["split"] == 0)
            ]
        elif split == DataSplit.VAL:
            # Validation: use specified validation centers
            filtered_df = metadata_df[metadata_df["center"].isin(self.val_centers)]
        elif split == DataSplit.TEST:
            # Test: use specified test centers
            filtered_df = metadata_df[metadata_df["center"].isin(self.test_centers)]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Create samples
        samples = []

        # Logic: We only apply 'shots' to the TRAIN split.
        # Validation and Test should be full for accurate evaluation.
        if self.shots == 0 or split != DataSplit.TRAIN:
            for _, row in filtered_df.iterrows():
                # Extract metadata
                patient_id = int(row["patient"])
                node_id = int(row["node"])
                x_coord = int(row["x_coord"])
                y_coord = int(row["y_coord"])
                label = int(row["tumor"])  # Binary: tumor (1) or non-tumor (0)

                # Construct image filename and path
                patient_str = f"{patient_id:03d}"
                image_filename = (
                    f"patch_patient_{patient_str}_node_{node_id}_"
                    f"x_{x_coord}_y_{y_coord}.png"
                )
                image_dir = (
                    self.root / "patches" / f"patient_{patient_str}_node_{node_id}"
                )
                image_path = image_dir / image_filename

                # Only include samples with existing image files
                if not image_path.exists():
                    continue

                samples.append(StandardSample(image_path=str(image_path), label=label))

            return samples

        # Few-Shot Logic: Stratified Random Sampling
        # Group candidates by class first
        # Fix seed for reproducibility
        random.seed(20)

        candidates_by_class = defaultdict(list)

        for _, row in filtered_df.iterrows():
            patient_id = int(row["patient"])
            node_id = int(row["node"])
            x_coord = int(row["x_coord"])
            y_coord = int(row["y_coord"])
            label = int(row["tumor"])

            patient_str = f"{patient_id:03d}"
            image_filename = (
                f"patch_patient_{patient_str}_node_{node_id}_"
                f"x_{x_coord}_y_{y_coord}.png"
            )
            image_dir = self.root / "patches" / f"patient_{patient_str}_node_{node_id}"
            image_path = image_dir / image_filename

            if image_path.exists():
                candidates_by_class[label].append(
                    StandardSample(image_path=str(image_path), label=label)
                )

        # Sample k from each class
        for label, candidates in candidates_by_class.items():
            count = min(self.shots, len(candidates))
            samples.extend(random.sample(candidates, count))

        return samples


@register_adapter("wbc_att")
class WBCAttAdapter(IDatasetAdapter):
    """
    Adapter for the WBC-Att (White Blood Cell Attributes) dataset.

    WBC-Att is a white blood cell classification dataset with:
    - train.csv, valid.csv, test.csv: Contains image paths and labels.
    - Images/: Directory containing image files.

    The adapter reads CSV files, maps string labels to integer indices,
    and constructs image paths for each sample.

    Supports few-shot learning scenarios where only N samples per class are used.
    """

    def __init__(self, root: str, shots: int = 0) -> None:
        """
        Initialize the WBC-Att adapter.

        Args:
            root: Root directory of WBC-Att dataset.
            shots: Number of samples per class (0 = full dataset).
        """
        self.root = Path(root)
        self.shots = shots

    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        """
        Load WBC-Att samples for the specified split.

        Args:
            split: The split to load (TRAIN, VAL, or TEST).

        Returns:
            List of StandardSample objects (image_path, label).
        """
        # Map DataSplit enum to CSV filename
        split_map = {
            DataSplit.TRAIN: "pbc_attr_v1_train.csv",
            DataSplit.VAL: "pbc_attr_v1_val.csv",
            DataSplit.TEST: "pbc_attr_v1_test.csv",
        }

        csv_path = self.root / split_map[split]
        image_base = self.root

        # Validate required files exist
        if not csv_path.exists():
            raise FileNotFoundError(f"{split_map[split]} not found at {csv_path}")
        if not image_base.exists():
            raise FileNotFoundError(f"images directory not found at {image_base}")

        # Load CSV
        csv_df = pd.read_csv(csv_path)

        # Create label mapping from string labels to integer indices
        unique_labels = sorted(csv_df["label"].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Create samples
        samples = []

        # Logic: We only apply 'shots' to the TRAIN split.
        # Validation and Test should be full for accurate evaluation.
        if self.shots == 0 or split != DataSplit.TRAIN:
            for _, row in csv_df.iterrows():
                # Get image path and label
                image_filename = row.get("path")
                if not image_filename:
                    continue

                image_path = image_base / image_filename
                label_str = row.get("label")
                label = label_to_idx[label_str]

                # Only include samples with existing image files
                if not image_path.exists():
                    continue

                samples.append(StandardSample(image_path=str(image_path), label=label))

            return samples

        # Few-Shot Logic: Stratified Random Sampling

        # Fix seed for reproducibility
        (42)

        candidates_by_class = defaultdict(list)

        for _, row in csv_df.iterrows():
            image_filename = row.get("path")
            if not image_filename:
                continue

            image_path = image_base / image_filename
            label_str = row.get("label")
            label = label_to_idx[label_str]

            if image_path.exists():
                candidates_by_class[label].append(
                    StandardSample(image_path=str(image_path), label=label)
                )

        # Sample k from each class
        for label, candidates in candidates_by_class.items():
            count = min(self.shots, len(candidates))
            samples.extend(random.sample(candidates, count))

        return samples


# --- JSON-based Dataset Adapters ---
# Generic adapter for datasets using JSON split files
# Each dataset has a split_<DATASET>.json file with format:
# {
#     "train": [[image_path, label, label_name], ...],
#     "val": [[image_path, label, label_name], ...],
#     "test": [[image_path, label, label_name], ...]
# }


class JsonDatasetAdapter(IDatasetAdapter):
    """
    Generic adapter for JSON-based biomedical datasets.

    This adapter works with datasets that define their splits in JSON files
    with the structure: {train, val, test}, each containing [image_path, label, label_name].

    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/<DATASET_NAME>/
    """

    def __init__(self, root: str, shots: int = 0) -> None:
        """
        Initialize the JSON dataset adapter.

        Args:
            root: Root directory of the dataset (contains images and split JSON).
            shots: Number of samples per class (0 = full dataset).
        """
        self.root = Path(root)
        self.shots = shots
        self.split_data = None

    def _load_split_data(self) -> dict:
        """Load split data from JSON file if not already loaded."""
        if self.split_data is None:
            # Find the JSON split file
            json_files = list(self.root.glob("split_*.json"))
            if not json_files:
                raise FileNotFoundError(f"No split_*.json file found in {self.root}")
            
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                self.split_data = json.load(f)
        
        return self.split_data

    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        """
        Load samples for the specified split from JSON.

        Args:
            split: The split to load (TRAIN, VAL, or TEST).

        Returns:
            List of StandardSample objects (image_path, label).
        """
        split_data = self._load_split_data()

        # Map DataSplit enum to JSON key
        split_map = {
            DataSplit.TRAIN: "train",
            DataSplit.VAL: "val",
            DataSplit.TEST: "test",
        }

        split_key = split_map[split]
        if split_key not in split_data:
            raise ValueError(f"Split '{split_key}' not found in JSON")

        split_samples = split_data[split_key]
        samples = []

        # Logic: We only apply 'shots' to the TRAIN split.
        # Validation and Test should be full for accurate evaluation.
        if self.shots == 0 or split != DataSplit.TRAIN:
            for item in split_samples:
                # Format: [image_path, label, label_name]
                image_path_rel = item[0]
                label = item[1]

                image_path = self.root / image_path_rel

                # Only include samples with existing image files
                if not image_path.exists():
                    continue

                samples.append(StandardSample(image_path=str(image_path), label=label))

            return samples

        # Few-Shot Logic: Stratified Random Sampling

        # Fix seed for reproducibility
        random.seed(42)

        candidates_by_class = defaultdict(list)

        for item in split_samples:
            image_path_rel = item[0]
            label = item[1]

            image_path = self.root / image_path_rel

            if image_path.exists():
                candidates_by_class[label].append(
                    StandardSample(image_path=str(image_path), label=label)
                )

        # Sample k from each class
        for label, candidates in candidates_by_class.items():
            count = min(self.shots, len(candidates))
            samples.extend(random.sample(candidates, count))

        return samples


@register_adapter("btmri")
class BTMRIAdapter(JsonDatasetAdapter):
    """
    Adapter for the BTMRI dataset (Brain Tumor MRI).

    Dataset structure:
    - split_BTMRI.json: Contains train/val/test splits
    - Images organized by tumor type: glioma_tumor/, meningioma_tumor/, pituitary_tumor/
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/BTMRI/
    """
    pass


@register_adapter("busi")
class BUSIAdapter(JsonDatasetAdapter):
    """
    Adapter for the BUSI dataset (Breast Ultrasound Images).

    Dataset structure:
    - split_BUSI.json: Contains train/val/test splits
    - Images organized by category: benign_tumor/, malignant_tumor/, normal/
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/BUSI/
    """
    pass


@register_adapter("chmnist")
class CHMNISTAdapter(JsonDatasetAdapter):
    """
    Adapter for the CHMNIST dataset (Colorectal Histology MNIST).

    Dataset structure:
    - split_CHMNIST.json: Contains train/val/test splits
    - Images organized by tissue type: adipose_tissue/, background/, connective_tissue/, etc.
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/CHMNIST/
    """
    pass


@register_adapter("covid_19")
class COVID19Adapter(JsonDatasetAdapter):
    """
    Adapter for the COVID-19 Chest X-Ray dataset.

    Dataset structure:
    - split_COVID_19.json: Contains train/val/test splits
    - Images organized by condition: covid_lungs/, normal_lungs/, viral_pneumonia_lungs/
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/COVID_19/
    """
    pass


@register_adapter("ctkidney")
class CTKidneyAdapter(JsonDatasetAdapter):
    """
    Adapter for the CT Kidney dataset.

    Dataset structure:
    - split_CTKidney.json: Contains train/val/test splits
    - Images organized by kidney condition
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/CTKidney/
    """
    pass


@register_adapter("dermamnist")
class DermaMNISTAdapter(JsonDatasetAdapter):
    """
    Adapter for the DermaMNIST dataset (Skin Lesion MNIST).

    Dataset structure:
    - split_DermaMNIST.json: Contains train/val/test splits
    - Images organized by skin lesion type
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/DermaMNIST/
    """
    pass


@register_adapter("kneexray")
class KneeXrayAdapter(JsonDatasetAdapter):
    """
    Adapter for the KneeXray dataset.

    Dataset structure:
    - split_KneeXray.json: Contains train/val/test splits
    - X-ray images of knees organized by condition
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/KneeXray/
    """
    pass


@register_adapter("kvasir")
class KvasirAdapter(JsonDatasetAdapter):
    """
    Adapter for the Kvasir dataset (Gastrointestinal Endoscopy).

    Dataset structure:
    - split_Kvasir.json: Contains train/val/test splits
    - Endoscopy images organized by finding type
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Kvasir/
    """
    pass


@register_adapter("lungcolon")
class LungColonAdapter(JsonDatasetAdapter):
    """
    Adapter for the LungColon dataset.

    Dataset structure:
    - split_LungColon.json: Contains train/val/test splits
    - Histopathology images of lung and colon tissues
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/LungColon/
    """
    pass


@register_adapter("octmnist")
class OCTMNISTAdapter(JsonDatasetAdapter):
    """
    Adapter for the OCTMNIST dataset (Optical Coherence Tomography MNIST).

    Dataset structure:
    - split_OCTMNIST.json: Contains train/val/test splits
    - OCT images of retinal conditions
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/OCTMNIST/
    """
    pass


@register_adapter("retina")
class RETINAAdapter(JsonDatasetAdapter):
    """
    Adapter for the RETINA dataset (Retinal Fundus Images).

    Dataset structure:
    - split_RETINA.json: Contains train/val/test splits
    - Retinal fundus images organized by condition
    
    Base path: /storage/projects3/e19-fyp-out-of-domain-gen-in-cv/RETINA/
    """
    pass
