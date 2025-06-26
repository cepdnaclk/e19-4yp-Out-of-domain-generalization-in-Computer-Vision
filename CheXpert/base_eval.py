import os
import csv
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import warnings

def load_biomedclip():
    """Load BiomedCLIP model with local checkpoint"""
    model_name = "biomedclip_local"
    CONFIG_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_config.json"
    WEIGHTS_PATH = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/e19-4yp-Out-of-domain-generalization-in-Computer-Vision/BioMedClip/checkpoints/open_clip_pytorch_model.bin"
    with open(CONFIG_PATH, "r") as f:
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
        pretrained=WEIGHTS_PATH,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    return model, preprocess, tokenizer, device

def get_similarity_scores(model, preprocess, tokenizer, device, image_paths, pathology_prompts):
    """Get similarity scores for multiple pathologies"""
    # Preprocess images
    images = []
    valid_paths = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(preprocess(img))
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        return None, []
    
    images = torch.stack(images).to(device)
    
    # Prepare all prompts
    all_prompts = []
    pathology_names = []
    for pathology, prompts in pathology_prompts.items():
        all_prompts.extend(prompts)
        pathology_names.append(pathology)
    
    # Tokenize all text prompts
    texts = tokenizer(all_prompts, context_length=256).to(device)
    
    # Get features
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine similarity)
        similarity = (100.0 * image_features @ text_features.T)
    
    # Reshape results to have per-pathology scores
    num_prompts_per_pathology = len(next(iter(pathology_prompts.values())))
    results = {}
    for i, pathology in enumerate(pathology_names):
        start_idx = i * num_prompts_per_pathology
        end_idx = (i + 1) * num_prompts_per_pathology
        pathology_similarity = similarity[:, start_idx:end_idx]
        results[pathology] = pathology_similarity.cpu().numpy()
    
    return results, valid_paths

def process_chexpert_data(csv_path, image_dir, output_dir, batch_size=32):
    """Process CheXpert dataset with BiomedCLIP"""
    # Load model
    model, preprocess, tokenizer, device = load_biomedclip()
    
    # Define pathology prompts (multiple prompts per pathology for robustness)
    pathology_prompts = {
        'No Finding': [
            "A normal chest radiograph with no abnormalities",
            "This chest X-ray shows no signs of disease",
            "Healthy chest radiograph without findings"
        ],
        'Cardiomegaly': [
            "Chest radiograph showing enlarged heart",
            "X-ray demonstrating cardiomegaly",
            "Cardiac enlargement visible on chest film"
        ],
        'Lung Opacity': [
            "Chest X-ray showing lung opacity",
            "Radiograph with pulmonary opacity",
            "Lung abnormality visible on chest film"
        ],
        # Add prompts for other pathologies similarly
        'Edema': [
            "Chest radiograph showing pulmonary edema",
            "X-ray demonstrating fluid in lungs",
            "Pulmonary edema visible on chest film"
        ],
        'Pleural Effusion': [
            "Chest X-ray showing pleural effusion",
            "Radiograph with fluid around the lungs",
            "Pleural effusion visible on chest film"
        ]
        # Add remaining pathologies...
    }
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output CSV
    output_csv = os.path.join(output_dir, 'chexpert_results.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['image_path', 'patient_id', 'study_id', 'view']
        for pathology in pathology_prompts:
            header.extend([f"{pathology}_gt", f"{pathology}_pred", f"{pathology}_prob"])
        writer.writerow(header)
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing CheXpert"):
        batch_df = df.iloc[i:i+batch_size]
        image_paths = [os.path.join(image_dir, p) for p in batch_df['Path']]
        
        # Get similarity scores for all pathologies
        similarity_results, valid_paths = get_similarity_scores(
            model, preprocess, tokenizer, device, 
            image_paths, pathology_prompts
        )
        
        if similarity_results is None:
            continue
        
        # Prepare batch results
        batch_results = []
        for j, img_path in enumerate(valid_paths):
            # Get original row data
            original_idx = i + j
            row = df.iloc[original_idx]
            
            # Extract patient/study info from path
            path_parts = row['Path'].split('/')
            patient_id = path_parts[2]  # patient00001
            study_id = path_parts[3]    # study1
            view = path_parts[4].split('_')[1]  # view1_frontal.jpg -> frontal
            
            # Prepare result row
            result_row = {
                'image_path': img_path,
                'patient_id': patient_id,
                'study_id': study_id,
                'view': view
            }
            
            # Add scores for each pathology
            for pathology in pathology_prompts:
                # Get ground truth (1.0, 0.0, or -1.0)
                gt = row[pathology]
                
                # Get prediction scores (average across prompts)
                pathology_scores = similarity_results[pathology][j]
                avg_score = pathology_scores.mean()
                prob = torch.sigmoid(torch.tensor(avg_score)).item()
                pred = 1 if prob > 0.5 else 0
                
                result_row.update({
                    f"{pathology}_gt": gt,
                    f"{pathology}_pred": pred,
                    f"{pathology}_prob": prob
                })
            
            batch_results.append(result_row)
        
        # Append to CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for result in batch_results:
                writer.writerow([result[key] for key in header])
    
    print(f"Results saved to {output_csv}")
    
    # Calculate and print metrics
    calculate_metrics(output_csv, pathology_prompts.keys())

def calculate_metrics(results_csv, pathologies):
    """Calculate and print evaluation metrics"""
    df = pd.read_csv(results_csv)
    
    for pathology in pathologies:
        # Filter out uncertain cases (-1) for evaluation
        eval_df = df[df[f'{pathology}_gt'].isin([0.0, 1.0])]
        
        if len(eval_df) == 0:
            print(f"\nNo valid ground truth cases for {pathology}")
            continue
        
        y_true = eval_df[f'{pathology}_gt']
        y_prob = eval_df[f'{pathology}_prob']
        y_pred = eval_df[f'{pathology}_pred']
        
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        
        print(f"\nMetrics for {pathology}:")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Configure paths
    csv_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/CheXpert-v1.0-small/train.csv"  # Path to CheXpert CSV
    image_dir = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/"  # Base directory for images
    output_dir = "chexpert_results"
    
    # Run processing
    process_chexpert_data(csv_path, image_dir, output_dir)