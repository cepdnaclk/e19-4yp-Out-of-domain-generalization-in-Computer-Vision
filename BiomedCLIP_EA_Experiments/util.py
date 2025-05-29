import torch
from torch.utils.data import Dataset, DataLoader

# 1. Paths & constants
METADATA_CSV = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"
PATCHES_DIR = "/home/E19_FYP_Domain_Gen_Data/patches"
CONFIG_PATH = "../BioMedClip/checkpoints/open_clip_config.json"
WEIGHTS_PATH = "../BioMedClip/checkpoints/open_clip_pytorch_model.bin"
MODEL_NAME = "biomedclip_local"
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiomedCLIPDataset(Dataset):
    def __init__(self, df, preprocess):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["tumor"].astype(int).tolist()
        self.preproc = preprocess

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
            img = self.preproc(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def append_filename_and_filepath(df):
    df["filename"] = df.apply(
        lambda r: f"patch_patient_{r.patient:03d}_node_{r.node}_x_{r.x_coord}_y_{r.y_coord}.png",
        axis=1
    )
    df["filepath"] = df.apply(
        lambda r: os.path.join(
            PATCHES_DIR,
            f"patient_{r.patient:03d}_node_{r.node}",
            r.filename
        ),
        axis=1
    )
    return df
