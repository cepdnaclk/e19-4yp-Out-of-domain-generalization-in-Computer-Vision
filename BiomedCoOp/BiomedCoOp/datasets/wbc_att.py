import os
import pandas as pd
import random
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class WBCAtt(DatasetBase):
    dataset_dir = "WBCAtt"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.image_base = root

        # CSV paths
        self.train_csv = os.path.join(root, "pbc_attr_v1_train.csv")
        self.val_csv = os.path.join(root, "pbc_attr_v1_val.csv")
        self.test_csv = os.path.join(root, "pbc_attr_v1_test.csv")

        # Few-shot settings
        self.num_shots = getattr(cfg.DATASET, "NUM_SHOTS", None)  # None = full dataset
        self.seed = getattr(cfg.DATASET, "SEED", 42)

        # Load label space from train split
        df_train = pd.read_csv(self.train_csv)
        unique_labels = sorted(df_train["label"].unique())
        self.all_class_names = list(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_class_names)}

        # Read full splits
        full_train = self.read_data(self.train_csv)
        full_val = self.read_data(self.val_csv)
        test = self.read_data(self.test_csv)

        if self.num_shots == 0:
            train, val = [], []
        elif self.num_shots is not None:
            train, val = self.create_few_shot_split(full_train, self.num_shots, self.seed)
        else:
            train, val = full_train, full_val


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, csv_path):
        items = []
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_path = os.path.join(self.image_base, row["path"])
            label_name = row["label"]
            label = self.label_to_idx[label_name]

            if os.path.exists(img_path):
                items.append(Datum(impath=img_path, label=label, classname=label_name))

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
                raise ValueError(
                    f"Not enough samples for class {cls_name} to create {num_shots}-shot dataset."
                )
            sampled_items = random.sample(items, num_shots)
            train_few.extend(sampled_items)
            val_few.extend(sampled_items)  # validation = same few-shot items

        return train_few, val_few
