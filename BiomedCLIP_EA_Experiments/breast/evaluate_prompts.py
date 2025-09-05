from typing import List
import util
import torch
import numpy as np
import os


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load breast dataset - using test split for evaluation
    test_features, test_labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="test",  
        cache_dir="./breast_cache"
    )
    print("Test embeddings extracted successfully.")

    # Convert to torch tensors
    test_features = torch.from_numpy(test_features)
    test_labels = torch.from_numpy(test_labels)

    general_negative_prompt = "A normal breast tissue image."
    general_positive_prompt = "A breast cancer image."
    
    # Optimized prompt pair (more specific medical terminology)
    optimized_negative_prompt = "Minimal posterior distortion"
    optimized_positive_prompt = "Significant posterior distortion"

    prompt_pairs = [
        ((general_negative_prompt, general_positive_prompt), "General"),
        ((optimized_negative_prompt, optimized_positive_prompt), "Optimized")
    ]

    print("="*80)
    print("BREAST CANCER DETECTION - PROMPT COMPARISON")
    print("="*80)
    
    results_comparison = {}
    
    for (negative_prompt, positive_prompt), prompt_type in prompt_pairs:
        print(f"\n{'-'*60}")
        print(f"Evaluating {prompt_type} Prompts:")
        print(f"{'-'*60}")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Positive Prompt: {positive_prompt}")
        print()
        
        results = util.evaluate_prompt_pair(
            negative_prompt, positive_prompt, test_features, test_labels, model, tokenizer)

        print(f"Results for {prompt_type} Prompts:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  ROC AUC: {results['auc']:.4f}")
        print(f"  Inverted BCE: {results['inverted_bce']:.4f}")
        print(f"  Confusion Matrix:\n{results['cm']}")
        print(f"  Classification Report:\n{results['report']}")
        
        # Store results for comparison
        results_comparison[prompt_type] = {
            'accuracy': results['accuracy'],
            'auc': results['auc'],
            'inverted_bce': results['inverted_bce'],
            'confusion_matrix': results['cm']
        }
        print()

    # Final comparison
    print("="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(f"General Prompts   - Accuracy: {results_comparison['General']['accuracy']:.4f}, AUC: {results_comparison['General']['auc']:.4f}, Inverted BCE: {results_comparison['General']['inverted_bce']:.4f}")
    print(f"Optimized Prompts - Accuracy: {results_comparison['Optimized']['accuracy']:.4f}, AUC: {results_comparison['Optimized']['auc']:.4f}, Inverted BCE: {results_comparison['Optimized']['inverted_bce']:.4f}")
    
    accuracy_improvement = results_comparison['Optimized']['accuracy'] - results_comparison['General']['accuracy']
    auc_improvement = results_comparison['Optimized']['auc'] - results_comparison['General']['auc']
    bce_improvement = results_comparison['Optimized']['inverted_bce'] - results_comparison['General']['inverted_bce']
    
    print(f"\nImprovement with Optimized Prompts:")
    print(f"  Accuracy improvement: {accuracy_improvement:+.4f}")
    print(f"  AUC improvement: {auc_improvement:+.4f}")
    print(f"  Inverted BCE improvement: {bce_improvement:+.4f}")


if __name__ == "__main__":
    main()
