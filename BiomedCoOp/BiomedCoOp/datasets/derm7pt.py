import os
import pandas as pd
import sys

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class Derm7pt(DatasetBase):
    dataset_dir = "derm7pt"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.images_dir = os.path.join(root, "images")

        # Paths to metadata and index files
        self.meta_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/meta.csv"
        self.train_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/train_indexes.csv"
        self.valid_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/valid_indexes.csv"
        self.test_idx_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/Derm7pt/release_v0/meta/test_indexes.csv"

        # Only melanoma binary task
        self.all_class_names = ["non-melanoma", "melanoma"]

        train = self.read_data(split="train")
        val = self.read_data(split="val")
        test = self.read_data(split="test")

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        items = []
        metadata = pd.read_csv(self.meta_path)

        # Load indexes based on split
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
            img_name = row["image"]
            # Binary label: 1 if melanoma, 0 otherwise
            label = int("melanoma" in str(row["diagnosis"]).lower())

            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path):
                cls_name = self.all_class_names[label]
                item = Datum(impath=img_path, label=label, classname=cls_name)
                items.append(item)

        return items
