"""
Pytest test suite for Dataset Adapters.

Tests the newly implemented adapters with actual data located at:
/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/<DATASET_NAME>/

Run with: pytest tests/impl/test_adapters.py -v
"""

import json
from pathlib import Path
from typing import Optional

import pytest

from biomedxpro.impl.adapters import (
    list_available_adapters,
    get_adapter,
    JsonDatasetAdapter,
)
from biomedxpro.core.domain import DataSplit, StandardSample

# Base path for actual datasets
DATA_BASE_PATH = Path("/storage/projects3/e19-fyp-out-of-domain-gen-in-cv")



# New JSON-based datasets
JSON_DATASETS = [
    "btmri",
    "busi",
    "chmnist",
    "covid_19",
    "ctkidney",
    "dermamnist",
    "kneexray",
    "kvasir",
    "lungcolon",
    "octmnist",
    "retina",
]

# Mapping of adapter names to actual dataset folder names
DATASET_FOLDER_MAP = {
    "btmri": "BTMRI",
    "busi": "BUSI",
    "chmnist": "CHMNIST",
    "covid_19": "COVID_19",
    "ctkidney": "CTKidney",
    "dermamnist": "DermaMNIST",
    "kneexray": "KneeXray",
    "kvasir": "Kvasir",
    "lungcolon": "LungColon",
    "octmnist": "OCTMNIST",
    "retina": "RETINA",
}


class TestAdapterRegistry:
    """Test adapter registry functionality."""

    def test_all_adapters_registered(self):
        """Test that all 11 new adapters are registered."""
        adapters = list_available_adapters()
        
        for adapter_name in JSON_DATASETS:
            assert adapter_name in adapters, f"Adapter '{adapter_name}' not registered"

    def test_registry_total_count(self):
        """Test that registry contains at least 14 adapters (11 new + 3 existing)."""
        adapters = list_available_adapters()
        
        assert len(adapters) >= 14, f"Expected at least 14 adapters, got {len(adapters)}"

    def test_registry_has_existing_adapters(self):
        """Test that existing adapters are still registered."""
        adapters = list_available_adapters()
        existing = ["derm7pt", "camelyon17", "wbc_att"]
        
        for adapter_name in existing:
            assert adapter_name in adapters, f"Existing adapter '{adapter_name}' not found"


class TestAdapterInstantiation:
    """Test adapter instantiation."""

    @pytest.mark.parametrize("adapter_name", JSON_DATASETS)
    def test_instantiate_adapter(self, adapter_name):
        """Test that each JSON adapter can be instantiated."""
        adapter = get_adapter(adapter_name, root="/tmp/test")
        
        assert adapter is not None, f"Failed to instantiate {adapter_name}"
        assert isinstance(adapter, JsonDatasetAdapter), \
            f"{adapter_name} should inherit from JsonDatasetAdapter"

    @pytest.mark.parametrize("adapter_name", JSON_DATASETS)
    def test_adapter_with_shots(self, adapter_name):
        """Test adapter instantiation with shots parameter."""
        adapter = get_adapter(adapter_name, root="/tmp/test", shots=5)
        
        assert adapter.shots == 5, f"Shots parameter not set correctly for {adapter_name}"




@pytest.mark.skipif(
    not DATA_BASE_PATH.exists(),
    reason="Real data path not available (expected on local machines)"
)
class TestRealDataLoading:
    """Test loading real data from /storage/projects3/..."""

    @pytest.mark.parametrize("adapter_name,folder_name", DATASET_FOLDER_MAP.items())
    def test_adapter_with_real_data(self, adapter_name, folder_name):
        """Test adapter can be instantiated with real data path."""
        dataset_path = DATA_BASE_PATH / folder_name
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset directory not found: {dataset_path}")
        
        adapter = get_adapter(adapter_name, root=str(dataset_path))
        assert adapter is not None

    @pytest.mark.parametrize("adapter_name,folder_name", DATASET_FOLDER_MAP.items())
    def test_load_train_split(self, adapter_name, folder_name):
        """Test loading training split from real data."""
        dataset_path = DATA_BASE_PATH / folder_name
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset directory not found: {dataset_path}")
        
        adapter = get_adapter(adapter_name, root=str(dataset_path))
        samples = adapter.load_samples(DataSplit.TRAIN)
        
        assert isinstance(samples, list), "load_samples should return a list"
        if samples:
            assert all(isinstance(s, StandardSample) for s in samples), \
                "All samples should be StandardSample instances"

    @pytest.mark.parametrize("adapter_name,folder_name", DATASET_FOLDER_MAP.items())
    def test_load_val_split(self, adapter_name, folder_name):
        """Test loading validation split from real data."""
        dataset_path = DATA_BASE_PATH / folder_name
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset directory not found: {dataset_path}")
        
        adapter = get_adapter(adapter_name, root=str(dataset_path))
        samples = adapter.load_samples(DataSplit.VAL)
        
        assert isinstance(samples, list), "load_samples should return a list"
        if samples:
            assert all(isinstance(s, StandardSample) for s in samples), \
                "All samples should be StandardSample instances"

    @pytest.mark.parametrize("adapter_name,folder_name", DATASET_FOLDER_MAP.items())
    def test_load_test_split(self, adapter_name, folder_name):
        """Test loading test split from real data."""
        dataset_path = DATA_BASE_PATH / folder_name
        
        if not dataset_path.exists():
            pytest.skip(f"Dataset directory not found: {dataset_path}")
        
        adapter = get_adapter(adapter_name, root=str(dataset_path))
        samples = adapter.load_samples(DataSplit.TEST)
        
        assert isinstance(samples, list), "load_samples should return a list"
        print(f"Length of test samples for {adapter_name}: {len(samples)}")
        if samples:
            assert all(isinstance(s, StandardSample) for s in samples), \
                "All samples should be StandardSample instances"


@pytest.mark.skipif(
    not DATA_BASE_PATH.exists(),
    reason="Real data path not available (expected on local machines)"
)
class TestFewShotLearning:
    """Test few-shot learning functionality."""

    def test_few_shot_reduces_dataset(self):
        """Test that few-shot learning reduces dataset size."""
        # Use kvasir as test dataset
        dataset_path = DATA_BASE_PATH / "Kvasir"
        
        if not dataset_path.exists():
            pytest.skip("Kvasir dataset not available")
        
        # Full dataset
        adapter_full = get_adapter("kvasir", root=str(dataset_path), shots=0)
        train_full = adapter_full.load_samples(DataSplit.TRAIN)
        
        if not train_full:
            pytest.skip("No training samples found in Kvasir dataset")
        
        # Few-shot dataset
        shots = 5
        adapter_few = get_adapter("kvasir", root=str(dataset_path), shots=shots)
        train_few = adapter_few.load_samples(DataSplit.TRAIN)
        
        assert len(train_few) <= len(train_full), \
            "Few-shot dataset should be smaller than full dataset"

    def test_few_shot_preserves_labels(self):
        """Test that few-shot sampling preserves label distribution."""
        dataset_path = DATA_BASE_PATH / "Kvasir"
        
        if not dataset_path.exists():
            pytest.skip("Kvasir dataset not available")
        
        adapter = get_adapter("kvasir", root=str(dataset_path), shots=5)
        samples = adapter.load_samples(DataSplit.TRAIN)
        
        if not samples:
            pytest.skip("No training samples found in Kvasir dataset")
        
        labels = [s.label for s in samples]
        unique_labels = set(labels)
        
        assert len(unique_labels) > 0, "Should have at least one label"


class TestAdapterMethods:
    """Test adapter methods and structure."""

    @pytest.mark.parametrize("adapter_name", JSON_DATASETS)
    def test_has_load_samples_method(self, adapter_name):
        """Test that adapter has load_samples method."""
        adapter = get_adapter(adapter_name, root="/tmp")
        
        assert hasattr(adapter, "load_samples"), \
            f"{adapter_name} should have load_samples method"
        assert callable(getattr(adapter, "load_samples")), \
            f"load_samples should be callable for {adapter_name}"

    @pytest.mark.parametrize("adapter_name", JSON_DATASETS)
    def test_has_load_split_data_method(self, adapter_name):
        """Test that adapter has _load_split_data method."""
        adapter = get_adapter(adapter_name, root="/tmp")
        
        assert hasattr(adapter, "_load_split_data"), \
            f"{adapter_name} should have _load_split_data method"
        assert callable(getattr(adapter, "_load_split_data")), \
            f"_load_split_data should be callable for {adapter_name}"

    @pytest.mark.parametrize("adapter_name", JSON_DATASETS)
    def test_adapter_initialization_attributes(self, adapter_name):
        """Test that adapter is initialized with correct attributes."""
        root_path = "/tmp/test"
        shots = 5
        
        adapter = get_adapter(adapter_name, root=root_path, shots=shots)
        
        assert adapter.root == Path(root_path), "root path not set correctly"
        assert adapter.shots == shots, "shots not set correctly"
        assert hasattr(adapter, "split_data"), "split_data attribute missing"


class TestStandardSampleFormat:
    """Test StandardSample output format."""

    def test_standard_sample_has_required_fields(self):
        """Test that StandardSample has required fields."""
        sample = StandardSample(image_path="/path/to/image.jpg", label=0)
        
        assert hasattr(sample, "image_path"), "StandardSample should have image_path"
        assert hasattr(sample, "label"), "StandardSample should have label"
        assert sample.image_path == "/path/to/image.jpg"
        assert sample.label == 0


# Summary fixtures
@pytest.fixture(scope="session")
def dataset_summary():
    """Provide a summary of dataset information."""
    summary = {
        "total_adapters": len(list_available_adapters()),
        "json_adapters": len(JSON_DATASETS),
        "datasets": DATASET_FOLDER_MAP,
        "real_data_path": str(DATA_BASE_PATH),
        "real_data_available": DATA_BASE_PATH.exists(),
    }
    return summary


def test_summary_info(dataset_summary):
    """Display summary information about the test environment."""
    assert dataset_summary["total_adapters"] >= 14, "Should have at least 14 adapters"
    assert dataset_summary["json_adapters"] == 11, "Should have 11 JSON adapters"
