import os
import pandas as pd
import sys
import random

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class NIHChestXray(DatasetBase):
    dataset_dir = "NIHChestXray"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.root_dir = os.path.join(root, "all_images")
        self.metadata_csv = os.path.join(root, "Data_Entry_2017.csv")
        self.train_val_list = os.path.join(root, "train_val_list.txt")
        self.test_list = os.path.join(root, "test_list.txt")

        # Binary classification: non-pneumonia vs pneumonia
        self.all_class_names = ["non-pneumonia", "pneumonia"]

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
            train, val = full_train, full_val


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        items = []
        metadata = pd.read_csv(self.metadata_csv)

        # Load split files
        if split in ["train", "val"]:
            with open(self.train_val_list, "r") as f:
                lines = f.read().splitlines()
            if split == "train":
                split_files = [l for i, l in enumerate(lines) if i % 5 != 0]  # 80% train
            else:
                split_files = [l for i, l in enumerate(lines) if i % 5 == 0]  # 20% val
        elif split == "test":
            with open(self.test_list, "r") as f:
                split_files = f.read().splitlines()
        else:
            raise ValueError(f"Unknown split: {split}")

        for _, row in metadata.iterrows():
            img_name = row["Image Index"]

            # Binary label: 1 = Pneumonia, 0 = Non-Pneumonia
            label = 1 if row["Finding Labels"] == "Pneumonia" else 0

            if img_name not in split_files:
                continue

            img_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(img_path):
                cls_name = self.all_class_names[label]
                items.append(Datum(impath=img_path, label=label, classname=cls_name))

        return items

    def create_few_shot_split(self, full_train, num_shots, seed):
        """
        Sample num_shots per class from full_train to create few-shot train and val.
        Validation uses the same few-shot samples (deterministic).
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
