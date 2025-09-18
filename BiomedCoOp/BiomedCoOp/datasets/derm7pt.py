import os
import pandas as pd
import sys
import random

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class Derm7pt(DatasetBase):
    dataset_dir = "derm7pt"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.images_dir = os.path.join(root, "images")

        # Metadata and split index files
        self.meta_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/meta.csv"
        self.train_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/train_indexes.csv"
        self.valid_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/valid_indexes.csv"
        self.test_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/test_indexes.csv"

        # Binary classification: melanoma vs non-melanoma
        self.all_class_names = ["non-melanoma", "melanoma"]

        # Few-shot settings
        self.num_shots = getattr(cfg.DATASET, "NUM_SHOTS", None)  # None = full dataset
        self.seed = getattr(cfg.DATASET, "SEED", 42)

        # Load splits
        full_train = self.read_data(split="train")
        full_val = self.read_data(split="val")
        test = self.read_data(split="test")

        
        if self.num_shots > 0:
            train, val = self.create_few_shot_split(full_train, self.num_shots, self.seed)
        else:
            print(f"{self.num_shots} Using full training set with samples.")
            train, val = full_train, full_val


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        items = []
        metadata = pd.read_csv(self.meta_path)

        # Select indices based on split
        if split == "train":
            indexes = pd.read_csv(self.train_idx_path)["indexes"].tolist()
        elif split == "val":
            indexes = pd.read_csv(self.valid_idx_path)["indexes"].tolist()
        elif split == "test":
            indexes = pd.read_csv(self.test_idx_path)["indexes"].tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

        metadata = metadata.iloc[indexes]

        for _, row in metadata.iterrows():
            img_name = row["derm"]
            label = int("melanoma" in str(row["diagnosis"]).lower())  # 1 = melanoma, 0 = non-melanoma
            img_path = os.path.join(self.images_dir, img_name)

            if os.path.exists(img_path):
                cls_name = self.all_class_names[label]
                items.append(Datum(impath=img_path, label=label, classname=cls_name))

        return items

    def create_few_shot_split(self, full_train, num_shots, seed):
        """
        Create few-shot train and val sets from full training data.
        - Train: num_shots per class
        - Val: same few-shot samples (deterministic)
        """
        random.seed(seed)
        train_few, val_few = [], []
        class_to_items = {cls_name: [] for cls_name in self.all_class_names}

        # Group by class
        for item in full_train:
            class_to_items[item.classname].append(item)

        # Sample num_shots per class
        for cls_name, items in class_to_items.items():
            if len(items) < num_shots:
                raise ValueError(f"Not enough samples for class {cls_name} to create {num_shots}-shot dataset.")
            sampled_items = random.sample(items, num_shots)
            train_few.extend(sampled_items)
            val_few.extend(sampled_items)  # validation = same few-shot items

        return train_few, val_few
