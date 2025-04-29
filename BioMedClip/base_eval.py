import os
import csv
import json
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# Load the model (same as before)
def load_biomedclip():
    # # Download the model and config files
    # hf_hub_download(
    #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     filename="open_clip_pytorch_model.bin",
    #     local_dir="checkpoints"
    # )
    # hf_hub_download(
    #     repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    #     filename="open_clip_config.json",
    #     local_dir="checkpoints"
    # )

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
    
    return model, preprocess, tokenizer, device

# Function to process images and get similarity scores
def get_similarity_scores(model, preprocess, tokenizer, device, image_paths, positive_prompt, negative_prompt):
    # Preprocess images
    images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(preprocess(img))
            valid_paths.append(img_path)
        except:
            continue
    
    if not images:
        return None, []
    
    images = torch.stack(images).to(device)
    
    # Tokenize text prompts
    texts = tokenizer([positive_prompt, negative_prompt], context_length=256).to(device)
    
    # Get features
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine similarity)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    return similarity.cpu().numpy(), valid_paths

# Function to extract coordinates from filename
def extract_coordinates(filename):
    # Example filename: "patient_004_node_4_x_3328_y_21792.png"
    parts = filename.split('_')
    try:
        x = int(parts[-3])
        y = int(parts[-1].split('.')[0])
        return x, y
    except:
        return None, None

# Main processing function
def process_camelyon_data(metadata_csv, patches_dir, output_csv, batch_size=32):
    # Load model
    model, preprocess, tokenizer, device = load_biomedclip()
    
    # Define prompts
    positive_prompt = "This is an image of a tumor"
    negative_prompt = "Tumor is not present in this image"
    
    # Read metadata and create coordinate lookup
    coord_to_truth = {}
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = int(row['x_coord'])
                y = int(row['y_coord'])
                coord_to_truth[(x, y)] = int(row['tumor'])
            except:
                continue
    
    # Prepare output
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['identifier', 'ground_truth', 'positive_score', 'negative_score', 'image_path'])
        
        # Process each patient folder
        patient_folders = [f for f in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, f))]
        
        for patient_folder in tqdm(patient_folders, desc="Processing patients"):
            patient_path = os.path.join(patches_dir, patient_folder)
            image_files = [f for f in os.listdir(patient_path) if f.endswith('.png') or f.endswith('.jpg')]
            
            # Process in batches
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i+batch_size]
                image_paths = [os.path.join(patient_path, f) for f in batch_files]
                
                # Get similarity scores
                similarity, valid_paths = get_similarity_scores(
                    model, preprocess, tokenizer, device, 
                    image_paths, positive_prompt, negative_prompt
                )
                
                if similarity is None:
                    continue
                
                # Write results for each image in batch
                for j, img_path in enumerate(valid_paths):
                    filename = os.path.basename(img_path)
                    x, y = extract_coordinates(filename)
                    
                    if x is None or y is None:
                        continue
                    
                    # Get ground truth (default to -1 if not in metadata)
                    ground_truth = coord_to_truth.get((x, y), -1)
                    
                    # Create identifier
                    identifier = f"{patient_folder}_x{x}_y{y}"
                    
                    # Write results
                    writer.writerow([
                        identifier,
                        ground_truth,
                        similarity[j][0],  # positive score
                        similarity[j][1],  # negative score
                        img_path
                    ])
                f.flush()  # Ensure data is written regularly

if __name__ == "__main__":
    # Configure paths
    metadata_csv = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"  # path to your metadata file
    patches_dir = "/home/E19_FYP_Domain_Gen_Data/patches"       # path to your patches directory
    output_csv = "biomedclip_results.csv"
    
    # Run processing
    process_camelyon_data(metadata_csv, patches_dir, output_csv)
    print(f"Results saved to {output_csv}")