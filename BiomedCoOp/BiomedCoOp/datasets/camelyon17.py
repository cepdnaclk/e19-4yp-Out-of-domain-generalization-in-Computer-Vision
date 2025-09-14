import os
import pandas as pd
import sys
import random

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class Camelyon17Custom(DatasetBase):
    dataset_dir = "camelyon17"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.metadata_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv"
        self.root_dir = root
        self.all_class_names = ["non-tumor", "tumor"]
        self.num_shots = getattr(cfg.DATASET, "NUM_SHOTS", None)  # None for full dataset
        self.seed = getattr(cfg.DATASET, "SEED", 42)

        # Load full splits first
        full_train = self.read_data(split="train")
        full_val = self.read_data(split="val")
        test = self.read_data(split="test")

        if self.num_shots is not None:
            # Few-shot mode
            train, val = self.create_few_shot_split(full_train, self.num_shots, self.seed)
        else:
            # Full dataset mode
            train = full_train
            val = full_val

        super().__init__(train_x=train, val=val, test=test)

    def create_few_shot_split(self, full_train, num_shots, seed):
        """
        Sample num_shots examples per class from full_train to create few-shot train and val.
        Validation uses the same few-shot samples.
        """
        random.seed(seed)  # ensure deterministic sampling
        train_few = []
        val_few = []
        class_to_items = {cls_name: [] for cls_name in self.all_class_names}

        # Group items by class
        for item in full_train:
            class_to_items[item.classname].append(item)

        # Sample few shots per class
        for cls_name, items in class_to_items.items():
            if len(items) < num_shots:
                raise ValueError(f"Not enough samples for class {cls_name} to create {num_shots}-shot dataset.")
            sampled_items = random.sample(items, num_shots)
            train_few.extend(sampled_items)
            val_few.extend(sampled_items)  # same items for validation

        return train_few, val_few
