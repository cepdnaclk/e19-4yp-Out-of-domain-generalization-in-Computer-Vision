# src/biomedxpro/impl/__init__.py
"""
Implementation Module: Concrete implementations of core interfaces.

This package provides concrete implementations of the abstract protocols
defined in biomedxpro.core.interfaces, including:
- Dataset adapters for multiple biomedical datasets
- The BiomedDataLoader for encoding and caching
- LLM clients for different model providers
- Mock implementations for testing
"""

from .adapters import (
    Camelyon17Adapter,
    Derm7ptAdapter,
    WBCAttAdapter,
    get_adapter,
    list_available_adapters,
    register_adapter,
)
from .data_loader import BiomedCLIPModel, BiomedDataLoader, ImagePathDataset
from .data_integration_example import (
    load_dataset_pipeline,
    load_multiple_splits,
)

__all__ = [
    # Adapters
    "Derm7ptAdapter",
    "Camelyon17Adapter",
    "WBCAttAdapter",
    "get_adapter",
    "register_adapter",
    "list_available_adapters",
    # Data Loader
    "BiomedDataLoader",
    "BiomedCLIPModel",
    "ImagePathDataset",
    # Integration
    "load_dataset_pipeline",
    "load_multiple_splits",
]
