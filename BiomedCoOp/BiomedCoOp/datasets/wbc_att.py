import os
import pandas as pd
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class WBCAtt(DatasetBase):
    dataset_dir = "WBCAtt"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.root_dir = root
        self.train_csv = os.path.join(root, "pbc_attr_v1_train.csv")
        self.val_csv = os.path.join(root, "pbc_attr_v1_val.csv")
        self.test_csv = os.path.join(root, "pbc_attr_v1_test.csv")
        self.image_base = os.path.join(root, "PBC_dataset_normal_DIB")

        # Read unique labels from train split
        df_train = pd.read_csv(self.train_csv)
        unique_labels = sorted(df_train["label"].unique())
        self.all_class_names = list(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_class_names)}

        # Build splits
        train = self.read_split(self.train_csv)
        val = self.read_split(self.val_csv)
        test = self.read_split(self.test_csv)

        super().__init__(train_x=train, val=val, test=test)

    def read_split(self, csv_path):
        items = []
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # image path is relative to PBC_dataset_normal_DIB
            img_path = os.path.join(self.image_base, row["path"])
            label_name = row["label"]
            label = self.label_to_idx[label_name]

            if os.path.exists(img_path):
                item = Datum(impath=img_path, label=label, classname=label_name)
                items.append(item)

        return items
