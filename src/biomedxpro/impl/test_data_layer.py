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
    WBCAttAdapter,
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
        assert "wbc_att" in adapters

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

    def test_load_samples_train_split(self):
        """Test loading training split from actual Derm7pt dataset."""
        derm7pt_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0")
        
        # Skip test if dataset not available
        if not (derm7pt_root / "meta" / "meta.csv").exists():
            pytest.skip("Derm7pt dataset not available")
        
        adapter = Derm7ptAdapter()
        samples = adapter.load_samples(str(derm7pt_root / "meta"), DataSplit.TRAIN)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_val_split(self):
        """Test loading validation split from actual Derm7pt dataset."""
        derm7pt_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0")
        
        if not (derm7pt_root / "meta" / "meta.csv").exists():
            pytest.skip("Derm7pt dataset not available")
        
        adapter = Derm7ptAdapter()
        samples = adapter.load_samples(str(derm7pt_root / "meta"), DataSplit.VAL)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_test_split(self):
        """Test loading test split from actual Derm7pt dataset."""
        derm7pt_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0")
        
        if not (derm7pt_root / "meta" / "meta.csv").exists():
            pytest.skip("Derm7pt dataset not available")
        
        adapter = Derm7ptAdapter()
        samples = adapter.load_samples(str(derm7pt_root / "meta"), DataSplit.TEST)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)


class TestCamelyon17Adapter:
    """Test the Camelyon17 adapter."""

    def test_load_samples_train_split(self):
        """Test loading training split from actual Camelyon17 dataset."""
        camelyon17_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS")
        
        # Skip test if dataset not available
        if not (camelyon17_root / "metadata.csv").exists():
            pytest.skip("Camelyon17 dataset not available")
        
        adapter = Camelyon17Adapter()
        samples = adapter.load_samples(str(camelyon17_root), DataSplit.TRAIN)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_val_split(self):
        """Test loading validation split from actual Camelyon17 dataset."""
        camelyon17_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS")
        
        if not (camelyon17_root / "metadata.csv").exists():
            pytest.skip("Camelyon17 dataset not available")
        
        adapter = Camelyon17Adapter()
        samples = adapter.load_samples(str(camelyon17_root), DataSplit.VAL)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_test_split(self):
        """Test loading test split from actual Camelyon17 dataset."""
        camelyon17_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS")
        
        if not (camelyon17_root / "metadata.csv").exists():
            pytest.skip("Camelyon17 dataset not available")
        
        adapter = Camelyon17Adapter()
        samples = adapter.load_samples(str(camelyon17_root), DataSplit.TEST)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)


class TestWBCAttAdapter:
    """Test the WBC-Att adapter."""

    def test_load_samples_train_split(self):
        """Test loading training split from actual WBC-Att dataset."""
        wbc_att_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att")
        
        # Skip test if dataset not available
        if not (wbc_att_root / "pbc_attr_v1_train.csv").exists():
            pytest.skip("WBC-Att dataset not available")
        
        adapter = WBCAttAdapter()
        samples = adapter.load_samples(str(wbc_att_root), DataSplit.TRAIN)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_val_split(self):
        """Test loading validation split from actual WBC-Att dataset."""
        wbc_att_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att")
        
        if not (wbc_att_root / "pbc_attr_v1_val.csv").exists():
            pytest.skip("WBC-Att dataset not available")
        
        adapter = WBCAttAdapter()
        samples = adapter.load_samples(str(wbc_att_root), DataSplit.VAL)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)
        assert isinstance(samples[0].label, int)

    def test_load_samples_test_split(self):
        """Test loading test split from actual WBC-Att dataset."""
        wbc_att_root = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/wbc_att")
        
        if not (wbc_att_root / "pbc_attr_v1_test.csv").exists():
            pytest.skip("WBC-Att dataset not available")
        
        adapter = WBCAttAdapter()
        samples = adapter.load_samples(str(wbc_att_root), DataSplit.TEST)

        assert len(samples) > 0
        assert isinstance(samples[0], StandardSample)


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
    def test_load_encoded_dataset_cache_hit(self, mock_model_class, tmp_path):
        """Test loading from cache."""
        # Pre-create a cache file with real tensors
        cache_key = "abcd1234"
        cache_filename = f"test_dataset_{cache_key}.pt"
        cache_path = tmp_path / cache_filename
        
        # Create REAL tensors (not mocks) for the cache
        cache_data = {
            "name": "test_dataset",
            "features": torch.randn(5, 512),
            "labels": torch.tensor([0, 1, 0, 1, 0]),
            "class_names": ["Class0", "Class1"],
        }
        torch.save(cache_data, cache_path)

        # Patch the cache key computation to return our known key
        with patch.object(BiomedDataLoader, "_compute_cache_key", return_value=cache_key):
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
            mock_model_class.assert_not_called()


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
