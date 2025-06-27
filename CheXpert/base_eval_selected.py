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

# Configuration - Easily modifiable
CONFIG = {
    "observations": [
        "Pneumonia"
    ],
    "batch_size": 32,
    "paths": {
        "config": "/path/to/open_clip_config.json",
        "weights": "/path/to/open_clip_pytorch_model.bin",
        "csv": "/path/to/CheXpert-v1.0-small/train.csv",
        "image_dir": "/path/to/images/",
        "output_dir": "chexpert_results"
    }
}

# Single prompt for each positive/negative case
PROMPTS = { 
    "Pneumonia": {
        "positive": "Chest radiograph showing pneumonia infection",
        "negative": "No signs of pneumonia"
    }
}

def load_biomedclip():
    """Load BiomedCLIP model with local checkpoint"""
    model_name = "biomedclip_local"

    with open(CONFIG["paths"]["config"], "r") as f:
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
        pretrained=CONFIG["paths"]["weights"],
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    return model, preprocess, tokenizer, device

def get_similarity_scores(model, preprocess, tokenizer, device, image_paths):
    """Get similarity scores for all defined observations"""
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
    prompt_types = []
    observation_names = []
    
    for obs in CONFIG["observations"]:
        all_prompts.extend([PROMPTS[obs]["positive"], PROMPTS[obs]["negative"]])
        prompt_types.extend(["positive", "negative"])
        observation_names.extend([obs, obs])
    
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
    
    # Organize results by observation
    results = {}
    for i, obs in enumerate(CONFIG["observations"]):
        pos_idx = i * 2
        neg_idx = pos_idx + 1
        results[obs] = {
            "positive": similarity[:, pos_idx].cpu().numpy(),
            "negative": similarity[:, neg_idx].cpu().numpy()
        }
    
    return results, valid_paths

def process_chexpert_data():
    """Process CheXpert dataset with BiomedCLIP"""
    # Load model
    model, preprocess, tokenizer, device = load_biomedclip()
    
    # Read CSV file
    df = pd.read_csv(CONFIG["paths"]["csv"])
    
    # Treat null values as negative (0.0)
    for obs in CONFIG["observations"]:
        df[obs] = df[obs].fillna(0.0)
    
    # Create output directory
    os.makedirs(CONFIG["paths"]["output_dir"], exist_ok=True)
    
    # Prepare output CSV
    output_csv = os.path.join(CONFIG["paths"]["output_dir"], 'chexpert_results.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        header = ['image_path', 'patient_id', 'study_id', 'view']
        for obs in CONFIG["observations"]:
            header.extend([
                f"{obs}_gt",
                f"{obs}_pred",
                f"{obs}_prob_pos",
                f"{obs}_prob_neg"
            ])
        writer.writerow(header)
    
    # Process in batches
    for i in tqdm(range(0, len(df), CONFIG["batch_size"]), desc="Processing CheXpert"):
        batch_df = df.iloc[i:i+CONFIG["batch_size"]]
        image_paths = [os.path.join(CONFIG["paths"]["image_dir"], p) for p in batch_df['Path']]
        
        # Get similarity scores
        similarity_results, valid_paths = get_similarity_scores(
            model, preprocess, tokenizer, device, image_paths
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
            
            # Add scores for each observation
            for obs in CONFIG["observations"]:
                # Get ground truth (1.0, 0.0, or -1.0)
                gt = row[obs]
                
                # Get prediction scores
                pos_score = similarity_results[obs]["positive"][j]
                neg_score = similarity_results[obs]["negative"][j]
                
                # Convert to probabilities using softmax
                pos_prob = torch.exp(torch.tensor(pos_score)) / (torch.exp(torch.tensor(pos_score)) + torch.exp(torch.tensor(neg_score)))
                neg_prob = 1 - pos_prob
                
                # Final prediction (positive if pos_prob > 0.5)
                pred = 1 if pos_prob > 0.5 else 0
                
                result_row.update({
                    f"{obs}_gt": gt,
                    f"{obs}_pred": pred,
                    f"{obs}_prob_pos": pos_prob.item(),
                    f"{obs}_prob_neg": neg_prob.item()
                })
            
            batch_results.append(result_row)
        
        # Append to CSV
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for result in batch_results:
                writer.writerow([result[key] for key in header])
    
    print(f"Results saved to {output_csv}")
    
    # Calculate and print metrics
    calculate_metrics(output_csv)

def calculate_metrics(results_csv):
    """Calculate and print evaluation metrics"""
    df = pd.read_csv(results_csv)
    
    for obs in CONFIG["observations"]:
        # Filter out uncertain cases (-1) for evaluation
        eval_df = df[df[f'{obs}_gt'].isin([0.0, 1.0])]
        
        if len(eval_df) == 0:
            print(f"\nNo valid ground truth cases for {obs}")
            continue
        
        y_true = eval_df[f'{obs}_gt']
        y_prob = eval_df[f'{obs}_prob_pos']  # Use positive probability for metrics
        y_pred = eval_df[f'{obs}_pred']
        
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        
        print(f"\nMetrics for {obs}:")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    process_chexpert_data()