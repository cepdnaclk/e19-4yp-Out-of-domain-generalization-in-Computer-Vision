# src/biomedxpro/impl/test_data_layer.py
"""
Test Suite: Data Layer Components

This module provides tests and validation for the data layer components.
It can be used to verify:
1. Adapter registration and retrieval
2. Sample loading from adapters
3. Caching behavior
4. Device handling

Run with: python -m pytest src/biomedxpro/impl/test_data_layer.py
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from biomedxpro.core.domain import DataSplit, StandardSample
from biomedxpro.impl.adapters import (
    Camelyon17Adapter,
    Derm7ptAdapter,
    ISICAdapter,
    get_adapter,
    list_available_adapters,
    register_adapter,
)
from biomedxpro.impl.data_loader import BiomedDataLoader


class TestAdapterRegistry:
    """Test the adapter registration and lookup system."""

    def test_list_available_adapters(self):
        """Test that registered adapters are discoverable."""
        adapters = list_available_adapters()
        assert isinstance(adapters, list)
        assert "derm7pt" in adapters
        assert "camelyon17" in adapters
        assert "isic" in adapters

    def test_get_adapter_success(self):
        """Test successful adapter retrieval."""
        adapter = get_adapter("derm7pt")
        assert isinstance(adapter, Derm7ptAdapter)

    def test_get_adapter_failure(self):
        """Test error handling for unknown adapter."""
        with pytest.raises(KeyError, match="not found"):
            get_adapter("unknown_adapter")

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        @register_adapter("test_adapter")
        class TestAdapter:
            def load_samples(self, root: str, split: DataSplit):
                return []

        adapter = get_adapter("test_adapter")
        assert isinstance(adapter, TestAdapter)


class TestDerm7ptAdapter:
    """Test the Derm7pt adapter."""

    @pytest.fixture
    def mock_derm7pt_structure(self, tmp_path):
        """Create a mock Derm7pt dataset structure."""
        # Create directories
        (tmp_path / "images").mkdir()

        # Create meta.csv
        meta_csv = tmp_path / "meta.csv"
        meta_csv.write_text(
            "diagnosis,derm\n"
            "melanoma,img1.jpg\n"
            "benign nevus,img2.jpg\n"
        )

        # Create split files
        train_csv = tmp_path / "train_indexes.csv"
        train_csv.write_text("indexes\n0\n")

        val_csv = tmp_path / "valid_indexes.csv"
        val_csv.write_text("indexes\n1\n")

        test_csv = tmp_path / "test_indexes.csv"
        test_csv.write_text("indexes\n0\n1\n")

        # Create dummy image files
        (tmp_path / "images" / "img1.jpg").write_bytes(b"fake image data")
        (tmp_path / "images" / "img2.jpg").write_bytes(b"fake image data")

        return tmp_path

    def test_load_samples_train_split(self, mock_derm7pt_structure):
        """Test loading training split."""
        adapter = Derm7ptAdapter()
        samples = adapter.load_samples(str(mock_derm7pt_structure), DataSplit.TRAIN)

        assert len(samples) == 1
        assert isinstance(samples[0], StandardSample)
        assert samples[0].label == 1  # melanoma

    def test_load_samples_val_split(self, mock_derm7pt_structure):
        """Test loading validation split."""
        adapter = Derm7ptAdapter()
        samples = adapter.load_samples(str(mock_derm7pt_structure), DataSplit.VAL)

        assert len(samples) == 1
        assert samples[0].label == 0  # benign

    def test_load_samples_missing_meta_csv(self, tmp_path):
        """Test error handling for missing meta.csv."""
        adapter = Derm7ptAdapter()

        with pytest.raises(FileNotFoundError, match="meta.csv"):
            adapter.load_samples(str(tmp_path), DataSplit.TRAIN)


class TestBiomedDataLoader:
    """Test the BiomedDataLoader."""

    def test_initialization(self):
        """Test loader initialization."""
        with tempfile.TemporaryDirectory() as cache_dir:
            loader = BiomedDataLoader(cache_dir=cache_dir, device="cpu")

            assert loader.cache_dir == Path(cache_dir)
            assert loader.device == torch.device("cpu")
            assert loader.batch_size == 32

    def test_cache_key_computation(self):
        """Test that sample order affects cache key."""
        loader = BiomedDataLoader()

        samples1 = [
            StandardSample(image_path="/path/img1.jpg", label=0),
            StandardSample(image_path="/path/img2.jpg", label=1),
        ]

        samples2 = [
            StandardSample(image_path="/path/img2.jpg", label=1),
            StandardSample(image_path="/path/img1.jpg", label=0),
        ]

        key1 = loader._compute_cache_key(samples1)
        key2 = loader._compute_cache_key(samples2)

        # Different order → different keys
        assert key1 != key2

    def test_cache_key_consistency(self):
        """Test that identical samples produce identical keys."""
        loader = BiomedDataLoader()

        samples = [
            StandardSample(image_path="/path/img1.jpg", label=0),
            StandardSample(image_path="/path/img2.jpg", label=1),
        ]

        key1 = loader._compute_cache_key(samples)
        key2 = loader._compute_cache_key(samples)

        assert key1 == key2

    def test_save_and_load_cache(self):
        """Test caching mechanism."""
        with tempfile.TemporaryDirectory() as cache_dir:
            loader = BiomedDataLoader(cache_dir=cache_dir, device="cpu")

            # Create a mock dataset
            from biomedxpro.core.domain import EncodedDataset

            dataset = EncodedDataset(
                name="test_dataset",
                features=torch.randn(10, 512),
                labels=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                class_names=["Class0", "Class1"],
            )

            # Save to cache
            cache_path = Path(cache_dir) / "test_cache.pt"
            loader._save_to_cache(dataset, cache_path)

            assert cache_path.exists()

            # Load from cache
            loaded_dataset = loader._load_from_cache(
                cache_path, "test_dataset", ["Class0", "Class1"]
            )

            assert loaded_dataset.num_samples == 10
            assert torch.allclose(dataset.features.cpu(), loaded_dataset.features.cpu())

    @patch("biomedxpro.impl.data_loader.BiomedCLIPModel")
    def test_load_encoded_dataset_cache_hit(self, mock_model, tmp_path):
        """Test loading from cache."""
        # Pre-create a cache file
        cache_path = tmp_path / "test_dataset_abcd1234.pt"
        cache_data = {
            "name": "test_dataset",
            "features": torch.randn(5, 512),
            "labels": torch.tensor([0, 1, 0, 1, 0]),
            "class_names": ["Class0", "Class1"],
        }
        torch.save(cache_data, cache_path)

        loader = BiomedDataLoader(cache_dir=str(tmp_path), device="cpu")
        samples = [
            StandardSample(image_path="/path/img1.jpg", label=0),
            StandardSample(image_path="/path/img2.jpg", label=1),
            StandardSample(image_path="/path/img3.jpg", label=0),
            StandardSample(image_path="/path/img4.jpg", label=1),
            StandardSample(image_path="/path/img5.jpg", label=0),
        ]

        # Load dataset (should hit cache, not call model)
        dataset = loader.load_encoded_dataset(
            "test_dataset",
            samples,
            ["Class0", "Class1"],
        )

        assert dataset.num_samples == 5
        # Model should NOT have been called (cache hit)
        mock_model.assert_not_called()


class TestIntegration:
    """Integration tests for the complete data pipeline."""

    def test_adapter_to_loader_pipeline(self):
        """Test the complete adapter → loader pipeline."""
        # This is a mock test since we need real dataset files
        # In production, this would use actual datasets

        adapter = get_adapter("derm7pt")
        assert adapter is not None

        loader = BiomedDataLoader(device="cpu")
        assert loader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
