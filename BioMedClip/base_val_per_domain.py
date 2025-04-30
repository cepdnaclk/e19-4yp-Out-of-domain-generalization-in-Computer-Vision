import os
import csv
import json
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def load_biomedclip():
    # Load the model and config files
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

def extract_patch_info(filename):
    """Extract patient, node, x, y from filename"""
    parts = filename.split('_')
    try:
        patient = int(parts[1])  # patient_004 -> 4
        node = int(parts[3])     # node_4 -> 4
        x = int(parts[-3])       # x_3328 -> 3328
        y = int(parts[-1].split('.')[0])  # y_21792.png -> 21792
        return patient, node, x, y
    except:
        return None, None, None, None

def process_camelyon_data(metadata_csv, patches_dir, output_dir, batch_size=32):
    # Load model
    model, preprocess, tokenizer, device = load_biomedclip()
    
    # Define prompts
    positive_prompt = "This is an image of a tumor"
    negative_prompt = "Tumor is not present in this image"
    
    # Read metadata and create composite key lookup
    metadata_lookup = {}
    with open(metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                patient = int(row['patient'])
                node = int(row['node'])
                x = int(row['x_coord'])
                y = int(row['y_coord'])
                key = (patient, node, x, y)
                metadata_lookup[key] = {
                    'tumor': int(row['tumor']),
                    'center': int(row['center'])
                }
            except:
                continue
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV writers for each center (0-4)
    center_writers = {}
    center_files = {}
    
    for center in range(5):
        center_csv = os.path.join(output_dir, f'center_{center}_results.csv')
        center_files[center] = open(center_csv, 'w', newline='')
        writer = csv.writer(center_files[center])
        writer.writerow(['identifier', 'ground_truth', 'positive_score', 'negative_score', 'image_path'])
        center_writers[center] = writer
    
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
                patient, node, x, y = extract_patch_info(filename)
                
                if None in (patient, node, x, y):
                    continue
                
                # Get ground truth and center info using composite key
                key = (patient, node, x, y)
                info = metadata_lookup.get(key, {'tumor': -1, 'center': -1})
                ground_truth = info['tumor']
                center = info['center']
                
                # Skip if center is invalid (not 0-4)
                if center not in center_writers:
                    continue
                
                # Create identifier
                identifier = f"{patient_folder}_x{x}_y{y}"
                
                # Write results to appropriate center file
                center_writers[center].writerow([
                    identifier,
                    ground_truth,
                    similarity[j][0],  # positive score
                    similarity[j][1],  # negative score
                    img_path
                ])
                center_files[center].flush()
    
    # Close all center files
    for center in center_files:
        center_files[center].close()
    
    print(f"Results saved to {output_dir} with separate files for each center")

if __name__ == "__main__":
    # Configure paths
    metadata_csv = "/home/E19_FYP_Domain_Gen_Data/metadata.csv"  
    patches_dir = "/home/E19_FYP_Domain_Gen_Data/patches"      
    output_dir = "biomedclip_results_by_center"
    
    # Run processing
    process_camelyon_data(metadata_csv, patches_dir, output_dir)