import json
import os
from glob import glob
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


CHECKPOINT_DIR = "./checkpoints"  
IMAGE_DIR = "./my_dataset/images" 
LABELS_PATH = "./my_dataset/labels.txt"  
MODEL_NAME = "biomedclip_local"  

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir=CHECKPOINT_DIR
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir=CHECKPOINT_DIR
)

with open(f"{CHECKPOINT_DIR}/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

if (not MODEL_NAME.startswith(HF_HUB_PREFIX)
    and MODEL_NAME not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[MODEL_NAME] = model_cfg

tokenizer = get_tokenizer(MODEL_NAME)
model, _, preprocess = create_model_and_transforms(
    model_name=MODEL_NAME,
    pretrained=f"{CHECKPOINT_DIR}/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

image_paths = glob(os.path.join(IMAGE_DIR, "*"))  # Supports .jpg, .png, etc.
if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}!")

images = torch.stack([
    preprocess(Image.open(path).convert("RGB")) 
    for path in image_paths
]).to(device)

template = "this is a photo of "
text_inputs = tokenizer(
    [template + label for label in labels],
    context_length=256
).to(device)

with torch.no_grad():
    image_features, text_features, logit_scale = model(images, text_inputs)
    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

logits = logits.cpu().numpy()
sorted_indices = sorted_indices.cpu().numpy()

print("\n===== Predictions =====")
for i, img_path in enumerate(image_paths):
    predicted_label = labels[sorted_indices[i][0]]
    confidence = logits[i][sorted_indices[i][0]]
    print(
        f"Image: {os.path.basename(img_path)}\n"
        f"  -> Predicted: {predicted_label} (Confidence: {confidence:.3f})\n"
    )

TOP_K = 3  # Show top 3 predictions per image
print(f"\n===== Top-{TOP_K} Predictions =====")
for i, img_path in enumerate(image_paths):
    print(f"Image: {os.path.basename(img_path)}")
    for k in range(TOP_K):
        idx = sorted_indices[i][k]
        print(f"  {k+1}. {labels[idx]} ({logits[i][idx]:.3f})")
    print()