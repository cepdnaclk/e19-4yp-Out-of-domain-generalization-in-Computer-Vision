import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import argparse
from tqdm import tqdm

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.dtype
        self.adapter = Adapter(512, 4).to(clip_model.text.transformer.dtype)
            
    def forward(self, image_features, text_features, alpha=0.2):
        x = self.adapter(image_features)
        image_features = alpha * x + (1 - alpha) * image_features
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features

def evaluate_checkpoint(
    checkpoint_path: str,
    clip_model: nn.Module,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    text_weights: torch.Tensor,
    classnames: list,
    alpha: float = 0.2,
    verbose: bool = True
):
    """
    Evaluate a saved CLIP-Adapter checkpoint on test data
    
    Args:
        checkpoint_path: Path to saved adapter weights
        clip_model: Original CLIP model
        test_features: Test image features [N, D]
        test_labels: Test labels [N]
        text_weights: Text features [C, D]
        classnames: List of class names
        alpha: Adapter mixing coefficient
        verbose: Whether to print detailed metrics
    """
    # Load model
    model = CustomCLIP(clip_model)
    model.adapter = torch.load(checkpoint_path)
    model.eval()
    model.cuda()
    
    # Move data to GPU
    test_features = test_features.cuda()
    text_weights = text_weights.cuda()
    
    # Run evaluation
    with torch.no_grad():
        logits = model(test_features, text_weights, alpha)
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    labels = test_labels.cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'macro_precision': precision_score(labels, preds, average='macro'),
        'macro_recall': recall_score(labels, preds, average='macro'),
        'macro_f1': f1_score(labels, preds, average='macro'),
        'weighted_precision': precision_score(labels, preds, average='weighted'),
        'weighted_recall': recall_score(labels, preds, average='weighted'),
        'weighted_f1': f1_score(labels, preds, average='weighted'),
        'confusion_matrix': confusion_matrix(labels, preds),
        'class_names': classnames,
        'predictions': preds,
        'probabilities': probs
    }
    
    # Per-class metrics
    if len(classnames) <= 20:  # Only show if not too many classes
        metrics['per_class'] = classification_report(
            labels, preds, target_names=classnames, output_dict=True
        )
    
    if verbose:
        print("\n" + "="*50)
        print(f"Evaluation Results for {checkpoint_path}")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=classnames))
        
        if len(classnames) <= 10:  # Only show CM for small number of classes
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
    
    return metrics

# Example usage:
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="caches/best_clipAdapterModel.pt" type=str, required=True, 
                       help="Path to saved adapter checkpoint")
    parser.add_argument("--test_features", type=str, required=True,
                       help="Path to test features .pt file")
    parser.add_argument("--test_labels", type=str, required=True,
                       help="Path to test labels .pt file")
    parser.add_argument("--text_weights", type=str, required=True,
                       help="Path to text weights .pt file")
    parser.add_argument("--classnames", type=str, nargs='+', required=True,
                       help="List of class names")
    parser.add_argument("--alpha", type=float, default=0.2,
                       help="Adapter mixing coefficient")
    args = parser.parse_args()
    
    # Load data
    test_features = torch.load(args.test_features)
    test_labels = torch.load(args.test_labels)