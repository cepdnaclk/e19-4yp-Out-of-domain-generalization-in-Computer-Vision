import os
import pandas as pd

import sys

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class Camelyon17Custom(DatasetBase):
    dataset_dir = "camelyon17"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        # Use absolute metadata path as specified
        self.metadata_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/metadata.csv"
        self.root_dir = root
        self.all_class_names = ["non-tumor", "tumor"]

        train = self.read_data(split="train")
        val = self.read_data(split="val")
        test = self.read_data(split="test")

        super().__init__(train_x=train, val=val, test=test)

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
            print(f"label is this {label}")
            img_name = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
            img_path = os.path.join(self.root_dir, f"patient_{patient}_node_{node}", img_name)
            print(f"Hi I am the img path {img_path}")
            if os.path.exists(img_path):
                print("Hii I am here.")
                cls_name = self.all_class_names[label]
                item = Datum(impath=img_path, label=label, classname=cls_name)
                items.append(item)
                print(f"Item {item}")
        return items
