# src/biomedxpro/taxonomy/dataset_slicer.py
"""
Dataset Slicer for Taxonomic Binary Views.

Provides zero-copy (or minimal-copy) filtering of multiclass datasets
into binary subsets for hierarchical node training.
"""

import torch
from loguru import logger

from biomedxpro.core.domain import EncodedDataset


class DatasetSlicer:
    """
    Static utility to slice a multiclass dataset into a binary (0 vs 1) view
    for a specific Decision Node.

    The View Manager: Creates lightweight binary datasets from parent multiclass data
    without copying the massive feature tensors (when possible).

    Key Operations:
    1. Map class names -> parent dataset indices
    2. Build boolean masks to identify relevant samples
    3. Remap labels: left_classes -> 0, right_classes -> 1
    4. Return new EncodedDataset with filtered features and remapped labels
    """

    @staticmethod
    def create_binary_view(
        dataset: EncodedDataset,
        left_classes: list[str],
        right_classes: list[str],
    ) -> EncodedDataset:
        """
        Creates a subset dataset where:
        - Samples in left_classes -> Label 0
        - Samples in right_classes -> Label 1
        - All other samples are ignored (filtered out)

        Args:
            dataset: The parent multiclass dataset to slice
            left_classes: Class names to map to binary label 0
            right_classes: Class names to map to binary label 1

        Returns:
            New EncodedDataset with:
                - features: Subset of parent features (zero-copy view if possible)
                - labels: Binary tensor (0s and 1s)
                - class_names: ["Left_Group", "Right_Group"]

        Raises:
            ValueError: If any class name is not found in parent dataset

        Example:
            Parent: ["Normal", "BCC", "Melanoma", "Nevus"] (indices 0-3)
            Left: ["BCC", "Melanoma"] -> maps to binary 0
            Right: ["Nevus"] -> maps to binary 1
            Result: Dataset with only samples from classes [1, 2, 3],
                    labels remapped to [0, 0, 1]
        """
        # 1. Map string class names to integer indices in the PARENT dataset
        # This allows us to find "Kidney Cyst" (index 5) in the parent tensor
        try:
            left_indices = [dataset.class_names.index(c) for c in left_classes]
            right_indices = [dataset.class_names.index(c) for c in right_classes]
        except ValueError as e:
            raise ValueError(
                f"Slicing Error: Class not found in parent dataset. "
                f"Available classes: {dataset.class_names}. Error: {e}"
            )

        # Validation: Ensure disjoint sets
        left_set = set(left_indices)
        right_set = set(right_indices)
        if left_set & right_set:
            raise ValueError(
                f"left_classes and right_classes must be disjoint. "
                f"Found overlap at indices: {left_set & right_set}"
            )

        # 2. Build Boolean Masks (The "View" Logic)
        # We find all rows where the label matches our target indices
        device = dataset.labels.device

        # Check against Left Set
        mask_left = torch.isin(
            dataset.labels, torch.tensor(left_indices, device=device)
        )

        # Check against Right Set
        mask_right = torch.isin(
            dataset.labels, torch.tensor(right_indices, device=device)
        )

        # Combined Mask (The subset of data we care about)
        total_mask = mask_left | mask_right

        num_selected = total_mask.sum().item()
        if num_selected == 0:
            logger.warning(
                f"Slice resulted in empty dataset! "
                f"Classes: {left_classes} vs {right_classes}"
            )

        # 3. Create the Binary Labels
        # We extract the relevant labels...
        relevant_labels = dataset.labels[total_mask]

        # ...and create a new tensor of 0s and 1s.
        # Initialize with 0 (Left). Then set Right indices to 1.
        binary_labels = torch.zeros_like(relevant_labels)

        # Identify which of the *kept* samples belong to the right group
        is_right = torch.isin(
            relevant_labels, torch.tensor(right_indices, device=device)
        )
        binary_labels[is_right] = 1

        # 4. Return the View
        # Notice we use advanced indexing on features.
        # PyTorch's tensor[mask] creates a new tensor (copy), but it's minimal
        # since we're only copying the subset, not the entire dataset.
        sliced_features = dataset.features[total_mask]

        logger.debug(
            f"Dataset slice created: {num_selected} samples "
            f"({mask_left.sum().item()} left, {mask_right.sum().item()} right) "
            f"from {dataset.num_samples} total"
        )

        return EncodedDataset(
            name=f"{dataset.name}_slice",
            features=sliced_features,  # Zero-copy view or minimal copy
            labels=binary_labels,
            # Semantics handled by TaskDefinition, these are just internal labels
            class_names=["Left_Group", "Right_Group"],
        )

    @staticmethod
    def validate_slice_coverage(
        dataset: EncodedDataset,
        left_classes: list[str],
        right_classes: list[str],
    ) -> dict[str, int]:
        """
        Validates that the slice will capture data and returns statistics.

        Useful for debugging empty slices or verifying tree construction.

        Args:
            dataset: The parent dataset
            left_classes: Classes for left branch
            right_classes: Classes for right branch

        Returns:
            Dictionary with statistics:
                - 'left_count': Number of samples in left_classes
                - 'right_count': Number of samples in right_classes
                - 'total_count': Total samples that will be in the slice
                - 'parent_count': Total samples in parent dataset

        Raises:
            ValueError: If class names not found in dataset
        """
        try:
            left_indices = [dataset.class_names.index(c) for c in left_classes]
            right_indices = [dataset.class_names.index(c) for c in right_classes]
        except ValueError as e:
            raise ValueError(
                f"Validation Error: Class not found. "
                f"Available: {dataset.class_names}. Error: {e}"
            )

        device = dataset.labels.device

        mask_left = torch.isin(
            dataset.labels, torch.tensor(left_indices, device=device)
        )
        mask_right = torch.isin(
            dataset.labels, torch.tensor(right_indices, device=device)
        )

        left_count = mask_left.sum().item()
        right_count = mask_right.sum().item()
        total_count = (mask_left | mask_right).sum().item()

        return {
            "left_count": int(left_count),
            "right_count": int(right_count),
            "total_count": int(total_count),
            "parent_count": int(dataset.num_samples),
        }
