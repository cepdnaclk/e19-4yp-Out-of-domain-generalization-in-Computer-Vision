import os
from queue import Full
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
        self.num_shots = cfg.DATASET.NUM_SHOTS  
        self.seed = getattr(cfg.DATASET, "SEED", 42)

        # Load full splits first
        full_train = self.read_data(split="train")
        full_val = self.read_data(split="val")
        test = self.read_data(split="test")
        print(f"HI {self.num_shots} ")
        if self.num_shots > 0:
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

    def read_data(self, split):
        items = []
        metadata = pd.read_csv(self.metadata_path)
        if split == "train":
            metadata = metadata[(metadata["center"].isin([0, 1, 2])) & (metadata["split"] == 0)]
        elif split == "val":
            metadata = metadata[metadata["center"] == 3]
        elif split == "test":
            metadata = metadata[metadata["center"] == 4]
        for _, row in metadata.iterrows():
            patient = f"{int(row['patient']):03d}"
            node = int(row['node'])
            x_coord = int(row['x_coord'])
            y_coord = int(row['y_coord'])
            label = int(row['tumor'])
            # print(f"label is this {label}")
            img_name = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
            img_path = os.path.join(self.root_dir, f"patches/patient_{patient}_node_{node}", img_name)
            # print(f"Hi I am the img path {img_path}")
            if os.path.exists(img_path):
                # print("Hii I am here.")
                cls_name = self.all_class_names[label]
                item = Datum(impath=img_path, label=label, classname=cls_name)
                items.append(item)
                # print(f"Item {item}")
        return items
