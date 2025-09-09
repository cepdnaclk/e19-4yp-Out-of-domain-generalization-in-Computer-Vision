from .camelyon17 import Camelyon17Custom  

dataset_list = {
    "Camelyon17": Camelyon17Custom
}

def build_dataset(cfg):
    """Return the dataset instance specified in cfg."""
    return dataset_list[cfg.DATASET.NAME](cfg)
