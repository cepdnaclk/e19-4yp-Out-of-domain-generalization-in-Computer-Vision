"""
Test cases for DatasetSlicer.

Tests the zero-copy filtering logic for converting multiclass datasets
into binary views for hierarchical node training.
"""

import pytest
import torch

from biomedxpro.core.domain import EncodedDataset
from biomedxpro.taxonomy.dataset_slicer import DatasetSlicer


class TestDatasetSlicer:
    """Test suite for DatasetSlicer binary view creation."""

    @pytest.fixture
    def sample_dataset(self) -> EncodedDataset:
        """
        Creates a toy multiclass dataset for testing.

        4 classes with 25 samples each (100 total):
        - Class 0: "Normal" (samples 0-24)
        - Class 1: "BCC" (samples 25-49)
        - Class 2: "Melanoma" (samples 50-74)
        - Class 3: "Nevus" (samples 75-99)
        """
        num_samples = 100
        num_features = 512

        # Create sequential labels for deterministic testing
        # 25 samples per class
        labels = torch.tensor([i // 25 for i in range(num_samples)])

        # Random features
        features = torch.randn(num_samples, num_features)

        return EncodedDataset(
            name="TestDataset",
            features=features,
            labels=labels,
            class_names=["Normal", "BCC", "Melanoma", "Nevus"],
        )

    def test_basic_binary_split(self, sample_dataset: EncodedDataset) -> None:
        """Test simple binary split: one class vs another."""
        # Split: Normal (0) vs BCC (1)
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["BCC"],
        )

        # Should have 50 samples (25 + 25)
        assert result.num_samples == 50
        assert result.num_classes == 2
        assert result.class_names == ["Left_Group", "Right_Group"]

        # Check binary labels
        assert result.labels.shape == (50,)
        # First 25 should be 0 (Normal -> Left)
        assert torch.all(result.labels[:25] == 0)
        # Next 25 should be 1 (BCC -> Right)
        assert torch.all(result.labels[25:] == 1)

        # Features should have correct shape
        assert result.features.shape == (50, 512)

    def test_multiclass_to_binary_group(self, sample_dataset: EncodedDataset) -> None:
        """Test grouping multiple classes into binary targets."""
        # Split: [Normal, BCC] vs [Melanoma, Nevus]
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal", "BCC"],
            right_classes=["Melanoma", "Nevus"],
        )

        # Should have all 100 samples
        assert result.num_samples == 100

        # First 50 should be labeled 0 (Normal + BCC)
        assert torch.all(result.labels[:50] == 0)
        # Last 50 should be labeled 1 (Melanoma + Nevus)
        assert torch.all(result.labels[50:] == 1)

    def test_partial_slice(self, sample_dataset: EncodedDataset) -> None:
        """Test slicing only a subset of classes."""
        # Split: Normal vs Melanoma (ignoring BCC and Nevus)
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["Melanoma"],
        )

        # Should have only 50 samples (25 Normal + 25 Melanoma)
        assert result.num_samples == 50

        # All labels should be binary
        assert torch.all((result.labels == 0) | (result.labels == 1))

        # Should have 25 of each
        assert (result.labels == 0).sum() == 25
        assert (result.labels == 1).sum() == 25

    def test_unbalanced_split(self, sample_dataset: EncodedDataset) -> None:
        """Test asymmetric split (1 class vs 3 classes)."""
        # Split: Normal vs [BCC, Melanoma, Nevus]
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["BCC", "Melanoma", "Nevus"],
        )

        assert result.num_samples == 100

        # 25 should be left (Normal)
        assert (result.labels == 0).sum() == 25
        # 75 should be right (others)
        assert (result.labels == 1).sum() == 75

    def test_invalid_class_name(self, sample_dataset: EncodedDataset) -> None:
        """Test that invalid class names raise ValueError."""
        with pytest.raises(ValueError, match="Class not found"):
            DatasetSlicer.create_binary_view(
                dataset=sample_dataset,
                left_classes=["Normal"],
                right_classes=["InvalidClass"],
            )

    def test_empty_intersection(self, sample_dataset: EncodedDataset) -> None:
        """Test that overlapping classes raise ValueError."""
        # This should be caught by DecisionNode validation, but test defense-in-depth
        with pytest.raises(ValueError, match="must be disjoint"):
            DatasetSlicer.create_binary_view(
                dataset=sample_dataset,
                left_classes=["Normal", "BCC"],
                right_classes=["BCC", "Melanoma"],  # BCC appears in both
            )

    def test_empty_result_warning(self, sample_dataset: EncodedDataset) -> None:
        """Test that slicing with non-existent data returns empty dataset."""
        # Create a dataset where no samples match the criteria
        # This is a pathological case but should handle gracefully
        empty_dataset = EncodedDataset(
            name="Empty",
            features=torch.randn(10, 512),
            labels=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),  # Only classes 0 and 1
            class_names=["Normal", "BCC", "Melanoma", "Nevus"],
        )

        # Try to slice classes that don't exist in the data
        result = DatasetSlicer.create_binary_view(
            dataset=empty_dataset,
            left_classes=["Melanoma"],  # Not present in labels
            right_classes=["Nevus"],  # Not present in labels
        )

        # Should return empty dataset
        assert result.num_samples == 0
        # Warning is logged but we don't test log capture (requires loguru config)

    def test_label_remapping_correctness(self, sample_dataset: EncodedDataset) -> None:
        """Test that label remapping maintains data integrity."""
        # Original labels: [0, 0, ..., 1, 1, ..., 2, 2, ..., 3, 3, ...]
        # Slice: Melanoma (2) vs Nevus (3)
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Melanoma"],  # Original index 2 -> Binary 0
            right_classes=["Nevus"],  # Original index 3 -> Binary 1
        )

        # Get the original indices of the sliced samples
        _ = sample_dataset.labels[50:]  # Last 50 samples (classes 2 and 3)

        # Verify mapping: original label 2 -> binary 0, original label 3 -> binary 1
        assert result.num_samples == 50
        assert torch.all(result.labels[:25] == 0)  # Melanoma
        assert torch.all(result.labels[25:] == 1)  # Nevus

    def test_feature_preservation(self, sample_dataset: EncodedDataset) -> None:
        """Test that feature vectors are correctly preserved during slicing."""
        # Slice a subset
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["BCC"],
        )

        # The first 25 features in result should match first 25 in original
        original_normal_features = sample_dataset.features[:25]
        assert torch.allclose(result.features[:25], original_normal_features)

        # The next 25 features should match original BCC features
        original_bcc_features = sample_dataset.features[25:50]
        assert torch.allclose(result.features[25:], original_bcc_features)

    def test_device_consistency(self, sample_dataset: EncodedDataset) -> None:
        """Test that slicing maintains device consistency (CPU/GPU)."""
        # Move to device (if CUDA available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_on_device = sample_dataset.to(device)

        result = DatasetSlicer.create_binary_view(
            dataset=dataset_on_device,
            left_classes=["Normal"],
            right_classes=["BCC"],
        )

        # Result should be on the same device
        assert result.features.device == device
        assert result.labels.device == device

    def test_validate_slice_coverage(self, sample_dataset: EncodedDataset) -> None:
        """Test the validation helper method."""
        stats = DatasetSlicer.validate_slice_coverage(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["BCC", "Melanoma"],
        )

        assert stats["left_count"] == 25  # Normal
        assert stats["right_count"] == 50  # BCC + Melanoma
        assert stats["total_count"] == 75
        assert stats["parent_count"] == 100

    def test_validate_slice_coverage_error(
        self, sample_dataset: EncodedDataset
    ) -> None:
        """Test that validation catches invalid class names."""
        with pytest.raises(ValueError, match="Class not found"):
            DatasetSlicer.validate_slice_coverage(
                dataset=sample_dataset,
                left_classes=["Normal"],
                right_classes=["InvalidClass"],
            )

    def test_order_independence(self, sample_dataset: EncodedDataset) -> None:
        """Test that class order doesn't affect the slice."""
        # Create two slices with different orderings
        result1 = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal", "BCC"],
            right_classes=["Melanoma", "Nevus"],
        )

        result2 = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["BCC", "Normal"],  # Swapped order
            right_classes=["Nevus", "Melanoma"],  # Swapped order
        )

        # Both should produce the same number of samples
        assert result1.num_samples == result2.num_samples

        # Label distribution should be identical
        assert (result1.labels == 0).sum() == (result2.labels == 0).sum()
        assert (result1.labels == 1).sum() == (result2.labels == 1).sum()

    def test_name_propagation(self, sample_dataset: EncodedDataset) -> None:
        """Test that dataset name is properly propagated with _slice suffix."""
        result = DatasetSlicer.create_binary_view(
            dataset=sample_dataset,
            left_classes=["Normal"],
            right_classes=["BCC"],
        )

        assert result.name == "TestDataset_slice"
