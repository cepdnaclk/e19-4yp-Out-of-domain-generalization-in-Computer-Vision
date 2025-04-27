import json
import os
from glob import glob
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# ====== MODEL SETUP (identical to original) ======
# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)

# Load the model and config files
model_name = "biomedclip_local"

with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# ====== CUSTOM DATASET SETUP (modified part) ======
# Configuration - modify these paths
dataset_dir = "./my_dataset"  # Your dataset folder
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    # Add/remove your custom classes here
]

# Alternative: Load labels from file
# with open(os.path.join(dataset_dir, "labels.txt")) as f:
#     labels = [line.strip() for line in f.readlines()]

# Load all images from directory
test_imgs = []
for ext in ['*.jpg', '*.png', '*.jpeg']:
    test_imgs.extend(glob(os.path.join(dataset_dir, ext)))

if not test_imgs:
    raise FileNotFoundError(f"No images found in {dataset_dir}!")

# ====== ZERO-SHOT CLASSIFICATION (same structure) ======
template = 'this is a photo of '
context_length = 256

# Preprocess images (same as original)
images = torch.stack([preprocess(Image.open(img).convert("RGB")) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)

with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)
    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

# ====== RESULTS (same output format) ======
top_k = 3  # Show top 3 predictions

for i, img_path in enumerate(test_imgs):
    img_name = os.path.basename(img_path)
    print(f"{img_name}:")
    
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f"  {labels[jth_index]}: {logits[i][jth_index]:.4f}")
    
    print()  # Newline between images