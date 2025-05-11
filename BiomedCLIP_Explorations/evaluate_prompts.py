import tokenize
import io
import re
import ast
from typing import List, Tuple
import os
from gemini import Gemini
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tqdm import tqdm
import heapq
from typing import Tuple, List, Optional
from prompt_learner import (
    BiomedCLIPDataset,
    append_filename_and_filepath,
    evaluate_prompt_pair,
    PriorityQueue,
    load_initial_prompts
)
from train_adapter import (Adapter, cosine_dist)


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')

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


def evaluate_prompt_pair_with_adapter(
    negative_prompt,
    positive_prompt,
    image_feats: torch.Tensor,    # (N, D), precomputed
    image_labels: torch.Tensor,   # (N,)
    model,
    tokenizer,
    adapter,
):

    # 2. Encode & adapt text prompts
    # --------------------------------
    text_inputs = tokenizer(
        [negative_prompt, positive_prompt],
        context_length=CONTEXT_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        text_feats = model.encode_text(text_inputs)          # (2, D)
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)
        text_feats = adapter(text_feats)                     # adapt
        text_feats = text_feats / text_feats.norm(dim=1, keepdim=True)

        logit_scale = model.logit_scale.exp()

        # move image feats to DEVICE for one matrix multiply
        feats = image_feats.to(DEVICE)                        # (N, D)
        labels = image_labels.to(DEVICE)

        # adapter network
        feats = adapter(feats)                     # adapt
        # (N,)
        feats = feats / feats.norm(dim=1, keepdim=True)

        # compute all logits at once: (N, 2)
        logits = logit_scale * (feats @ text_feats.t())
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 0].cpu().numpy()    # tumor-class prob
        y_true = labels.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    return {'accuracy': acc, 'auc': auc, 'cm': cm, 'report': report}


def main():
    # 1. Load model and preprocess
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    model_cfg, preproc_cfg = cfg["model_cfg"], cfg["preprocess_cfg"]

    # register local config if needed
    if (not MODEL_NAME.startswith(HF_HUB_PREFIX)
            and MODEL_NAME not in _MODEL_CONFIGS):
        _MODEL_CONFIGS[MODEL_NAME] = model_cfg

    tokenizer = get_tokenizer(MODEL_NAME)
    model, _, preprocess = create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=WEIGHTS_PATH,
        **{f"image_{k}": v for k, v in preproc_cfg.items()}
    )

    model.to(DEVICE).eval()

    random_state = 42
    # samples = 2048

    # 2. Load metadata and filter center=0
    metadata_df = pd.read_csv(METADATA_CSV, index_col=0)
    metadata_df = append_filename_and_filepath(metadata_df)

    df = metadata_df[metadata_df.split == 1].copy()
    centers_ds = []
    for i in range(4):
        centers_ds.append(BiomedCLIPDataset(df[df.center == i], preprocess))

    # 3. Generate embeddings and store them for each center
    centers_features = [[] for _ in range(len(centers_ds))]
    centers_labels = [[] for _ in range(len(centers_ds))]

    for i, ds in enumerate(centers_ds):
        print(f"Extracting features for center {i}...")

        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Center {i}", unit="batch"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())

        centers_features[i] = torch.cat(all_feats)
        centers_labels[i] = torch.cat(all_labels)

    # load adapter
    adapter = Adapter(dim=512).to(DEVICE)
    adapter.load_state_dict(torch.load(
        "adapter_weights.pth", map_location=DEVICE))
    adapter.eval()

    initial_list = load_initial_prompts("selected_prompts.txt")
    pq = PriorityQueue(max_capacity=40, initial=initial_list)
    (negative_prompt, positive_prompt), score = pq.get_best()

    print(
        f"Best prompts: {negative_prompt}, {positive_prompt} with score: {score}")

    # 4. Evaluate the best prompts
    print("Evaluating best prompts...")

    for i, ds in enumerate(centers_ds):
        print(f"Evaluating center {i}...")
        print("Without adapter:")
        print(evaluate_prompt_pair(
            negative_prompt, positive_prompt, centers_features[i], centers_labels[i], model, tokenizer)
        )

        print("With adapter:")
        print(evaluate_prompt_pair_with_adapter(
            negative_prompt, positive_prompt, centers_features[i], centers_labels[i], model, tokenizer, adapter)
        )


if __name__ == "__main__":
    main()
