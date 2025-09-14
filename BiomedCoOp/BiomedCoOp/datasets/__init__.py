from .camelyon17 import Camelyon17Custom 
from .derm7pt import Derm7pt
from .wbc_att import WBCAtt

dataset_list = {
    "Camelyon17": Camelyon17Custom,
    "Derm7pt": Derm7pt,
    "WBCATT": WBCAtt
}

def build_dataset(cfg):
    """Return the dataset instance specified in cfg."""
    return dataset_list[cfg.DATASET.NAME](cfg)
