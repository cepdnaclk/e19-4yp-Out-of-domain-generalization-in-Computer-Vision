# src/biomedxpro/impl/data_integration_example.py
"""
Example: Complete Data Layer Integration

This module demonstrates how all data layer components work together:
1. Configuration loads adapter name from config
2. Registry returns the appropriate adapter instance
3. Adapter loads and standardizes samples for a specific dataset
4. DataLoader processes and encodes the samples
5. EncodedDataset is passed to the evolutionary engine

This is a reference implementation showing the typical workflow.
Production code should integrate this into the main orchestrator.
"""

from pathlib import Path
from typing import Optional

from biomedxpro.core.domain import DataSplit, EncodedDataset
from biomedxpro.impl.adapters import get_adapter, list_available_adapters
from biomedxpro.impl.data_loader import BiomedDataLoader


def load_dataset_pipeline(
    dataset_config: dict,
    cache_dir: str = ".biomedxpro_cache",
    device: Optional[str] = None,
) -> EncodedDataset:
    """
    Complete data pipeline: Configuration → Adapter → Loader → EncodedDataset
    
    Configuration Format (example):
    {
        "adapter": "derm7pt",           # Which adapter to use
        "root": "/path/to/derm7pt",     # Dataset root directory
        "split": "TRAIN",               # Which split (TRAIN, VAL, TEST)
        "name": "derm7pt_train",        # Name for this dataset
        "class_names": ["Benign", "Malignant"],  # Class names
        "few_shot": False,              # Enable few-shot learning (optional)
        "few_shot_no": 2                # Samples per class in few-shot (optional)
    }
    
    Args:
        dataset_config: Configuration dictionary (see format above).
        cache_dir: Directory for encoded dataset cache.
        device: Device for processing (cuda/cpu). Auto-detect if None.
    
    Returns:
        An EncodedDataset ready for the evolutionary engine.
    
    Raises:
        KeyError: If adapter name is not registered.
        FileNotFoundError: If dataset files are missing.
        ValueError: If configuration is incomplete.
    
    Example:
        >>> config = {
        ...     "adapter": "derm7pt",
        ...     "root": "/data/derm7pt",
        ...     "split": "TRAIN",
        ...     "name": "derm7pt_train",
        ...     "class_names": ["Benign", "Malignant"],
        ...     "few_shot": True,
        ...     "few_shot_no": 5
        ... }
        >>> dataset = load_dataset_pipeline(config)
        >>> print(f"Loaded {dataset.num_samples} samples")
    """
    # Validate configuration
    required_keys = ["adapter", "root", "split", "name", "class_names"]
    missing = [k for k in required_keys if k not in dataset_config]
    if missing:
        raise ValueError(
            f"Missing required configuration keys: {missing}. "
            f"Available adapters: {list_available_adapters()}"
        )

    # Extract configuration
    adapter_name = dataset_config["adapter"]
    root = dataset_config["root"]
    split = DataSplit(dataset_config["split"].upper())
    name = dataset_config["name"]
    class_names = dataset_config["class_names"]
    few_shot = dataset_config.get("few_shot", False)
    few_shot_no = dataset_config.get("few_shot_no", 2)

    # Step 1: Get the adapter
    print(f"[1/3] Selecting adapter: {adapter_name}")
    adapter = get_adapter(adapter_name)
    
    # Initialize adapter with few-shot parameters if supported
    try:
        adapter = get_adapter(adapter_name)
        # Check if adapter supports few-shot by trying to initialize with parameters
        import inspect
        sig = inspect.signature(adapter.__class__.__init__)
        if 'few_shot' in sig.parameters:
            # Reinitialize adapter with few-shot parameters
            adapter = adapter.__class__(few_shot=few_shot, few_shot_no=few_shot_no)
    except Exception:
        # If initialization fails, use default adapter
        adapter = get_adapter(adapter_name)

    # Step 2: Load samples using the adapter
    print(f"[2/3] Loading samples from {root}...")
    samples = adapter.load_samples(root, split)
    print(f"      Found {len(samples)} samples")

    # Step 3: Encode samples and return dataset
    print(f"[3/3] Processing and encoding samples...")
    loader = BiomedDataLoader(cache_dir=cache_dir, device=device)
    encoded_dataset = loader.load_encoded_dataset(name, samples, class_names)

    print(f"✓ Loaded EncodedDataset: {name}")
    print(f"  - Samples: {encoded_dataset.num_samples}")
    print(f"  - Classes: {encoded_dataset.num_classes} {encoded_dataset.class_names}")
    print(f"  - Features shape: {encoded_dataset.features.shape}")
    if few_shot:
        print(f"  - Few-shot mode: {few_shot_no} samples per class")

    return encoded_dataset


def load_multiple_splits(
    dataset_configs: list[dict],
    cache_dir: str = ".biomedxpro_cache",
    device: Optional[str] = None,
) -> dict[str, EncodedDataset]:
    """
    Load multiple dataset splits (e.g., train, val, test).
    
    This is a convenience function for loading an entire dataset
    with multiple splits at once.
    
    Args:
        dataset_configs: List of configuration dictionaries,
                        one per split.
        cache_dir: Directory for cache.
        device: Device for processing.
    
    Returns:
        Dictionary mapping dataset names to EncodedDataset objects.
    
    Example:
        >>> configs = [
        ...     {
        ...         "adapter": "derm7pt",
        ...         "root": "/data/derm7pt",
        ...         "split": "TRAIN",
        ...         "name": "derm7pt_train",
        ...         "class_names": ["Benign", "Malignant"]
        ...     },
        ...     {
        ...         "adapter": "derm7pt",
        ...         "root": "/data/derm7pt",
        ...         "split": "VAL",
        ...         "name": "derm7pt_val",
        ...         "class_names": ["Benign", "Malignant"]
        ...     }
        ... ]
        >>> datasets = load_multiple_splits(configs)
        >>> print(datasets.keys())  # dict_keys(['derm7pt_train', 'derm7pt_val'])
    """
    datasets = {}
    for config in dataset_configs:
        dataset = load_dataset_pipeline(config, cache_dir, device)
        datasets[config["name"]] = dataset

    return datasets


if __name__ == "__main__":
    # Example usage: Load Derm7pt training split
    config = {
        "adapter": "derm7pt",
        "root": "/path/to/derm7pt",
        "split": "TRAIN",
        "name": "derm7pt_train",
        "class_names": ["Benign", "Malignant"],
    }

    # This would load the dataset (if the path exists)
    # dataset = load_dataset_pipeline(config)
    print("Data layer integration example loaded successfully.")
    print(f"Available adapters: {list_available_adapters()}")
