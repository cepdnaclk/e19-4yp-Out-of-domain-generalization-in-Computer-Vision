import os
import pandas as pd
import sys

sys.path.append("BiomedCoOp/BiomedCoOp/Dassl.pytorch")

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class NIHChestXray(DatasetBase):
    dataset_dir = "nih_chest"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.root_dir = root
        self.metadata_csv = os.path.join(root, "Data_Entry_2017.csv")
        self.train_val_list = os.path.join(root, "train_val_list.txt")
        self.test_list = os.path.join(root, "test_list.txt")
        
        # Binary classification: non-pneumonia vs pneumonia
        self.all_class_names = ["Non-pneumonia", "Pneumonia"]

        train = self.read_data(split="train")
        val = self.read_data(split="val")
        test = self.read_data(split="test")

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

        for _, row in metadata.iterrows():
            img_name = row["Image Index"]
            
            # Binary label: 0 = Non-Pneumonia, 1 = Pneumonia
            label = 0 if row["Finding Labels"] == "No Finding" else 1

            if img_name not in split_files:
                continue

            # Determine folder from filename prefix (images_001, images_002, etc.)
            folder_idx = int(img_name.split("_")[0]) % 12 + 1
            folder_name = f"images_{folder_idx:03d}"
            img_path = os.path.join(self.root_dir, folder_name, img_name)

            if os.path.exists(img_path):
                cls_name = self.all_class_names[label]
                item = Datum(impath=img_path, label=label, classname=cls_name)
                items.append(item)

        return items
