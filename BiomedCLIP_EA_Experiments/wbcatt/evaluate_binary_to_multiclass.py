#!/usr/bin/env python3
"""
Binary to Multiclass Evaluation Script for WBC Classification

This script loads binary classifiers (one-vs-rest) for each WBC class from crowded results
and combines them into a multiclass ensemble for evaluation on the test set.

The process:
1. Load crowded binary prompt pairs for each WBC class
2. For each test sample, get binary predictions from all class-specific ensembles
3. Combine binary predictions into multiclass predictions
4. Evaluate multiclass performance

Usage:
    python evaluate_binary_to_multiclass.py --few_shot N
    
Example:
    python evaluate_binary_to_multiclass.py --few_shot 1
"""

import argparse
import os
import numpy as np
import torch
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

import util


def load_crowded_prompts_for_class(few_shot: int, class_label: str) -> List[Tuple[Tuple[str, str], float]]:
    """
    Load crowded prompt pairs for a specific class.

    Args:
        few_shot: Few-shot configuration (1, 2, 4, 8)
        class_label: WBC class name

    Returns:
        List of (prompt_pair, score) tuples
    """
    filename = f"final_results/crowded/{few_shot}-shot-{class_label}.txt"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Crowded prompts file not found: {filename}")

    prompts = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse line format: "('neg_prompt', 'pos_prompt'), Score: 0.xxxx"
            try:
                # Split by ", Score: "
                prompt_part, score_part = line.rsplit(", Score: ", 1)
                score = float(score_part)

                # Parse the prompt tuple
                # Remove outer parentheses and quotes, then split
                prompt_part = prompt_part.strip("()")
                neg_prompt, pos_prompt = prompt_part.split("', '")
                neg_prompt = neg_prompt.strip("'")
                pos_prompt = pos_prompt.strip("'")

                prompts.append(((neg_prompt, pos_prompt), score))

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line in {filename}: {line}")
                continue

    print(f"Loaded {len(prompts)} crowded prompts for {class_label}")
    return prompts


def evaluate_binary_ensemble(
    prompt_list: List[Tuple[Tuple[str, str], float]],
    image_feats: torch.Tensor,
    model,
    tokenizer,
    class_name: str
) -> np.ndarray:
    """
    Evaluate binary ensemble for a specific class (one-vs-rest).

    Args:
        prompt_list: List of (prompt_pair, score) tuples
        image_feats: Image features tensor (N, D)
        model: CLIP model
        tokenizer: CLIP tokenizer
        class_name: Name of the positive class

    Returns:
        Binary probabilities for the positive class (N,)
    """
    if not prompt_list:
        print(f"Warning: No prompts for {class_name}, returning zeros")
        return np.zeros(len(image_feats))

    all_weighted_probs = []
    total_weight = 0.0

    feats = image_feats.to(util.DEVICE)

    print(f"Evaluating {len(prompt_list)} binary prompts for {class_name}...")

    with torch.no_grad():
        for (neg_prompt, pos_prompt), weight in prompt_list:
            # Tokenize the binary prompt pair
            text_inputs = tokenizer(
                [neg_prompt, pos_prompt],
                context_length=util.CONTEXT_LENGTH
            ).to(util.DEVICE)

            # Get text features
            text_feats = model.encode_text(text_inputs)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (feats @ text_feats.T)  # (N, 2)

            # Convert to probabilities (softmax)
            probs = torch.softmax(similarities, dim=-1)  # (N, 2)

            # Get probability for positive class (index 1)
            pos_probs = probs[:, 1].cpu().numpy()  # (N,)

            # Weight the probabilities
            weighted_probs = pos_probs * weight
            all_weighted_probs.append(weighted_probs)
            total_weight += weight

    if total_weight == 0:
        print(
            f"Warning: Total weight is 0 for {class_name}, returning uniform probabilities")
        return np.full(len(image_feats), 0.5)

    # Ensemble the weighted probabilities
    ensemble_probs = np.sum(all_weighted_probs, axis=0) / total_weight

    return ensemble_probs


def convert_binary_to_multiclass(binary_probs_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert binary one-vs-rest probabilities to multiclass predictions.

    Args:
        binary_probs_dict: Dictionary mapping class_name -> binary_probabilities (N,)

    Returns:
        multiclass_probs: Probabilities for each class (N, num_classes)
        multiclass_preds: Predicted class indices (N,)
    """
    class_names = util.CLASSES
    num_samples = len(next(iter(binary_probs_dict.values())))
    num_classes = len(class_names)

    # Stack binary probabilities
    multiclass_probs = np.zeros((num_samples, num_classes))

    for i, class_name in enumerate(class_names):
        if class_name in binary_probs_dict:
            multiclass_probs[:, i] = binary_probs_dict[class_name]
        else:
            print(f"Warning: No probabilities for {class_name}, using zeros")
            multiclass_probs[:, i] = 0.0

    # Normalize probabilities (optional - makes it a proper probability distribution)
    row_sums = multiclass_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    multiclass_probs = multiclass_probs / row_sums

    # Get predictions (argmax)
    multiclass_preds = np.argmax(multiclass_probs, axis=1)

    return multiclass_probs, multiclass_preds


def main():
    parser = argparse.ArgumentParser(
        description="Binary to Multiclass Evaluation for WBC Classification")
    parser.add_argument("--few_shot", type=int, required=True,
                        choices=[1, 2, 4, 8],
                        help="Few-shot configuration to evaluate")

    args = parser.parse_args()

    print(
        f"=== Binary to Multiclass Evaluation for {args.few_shot}-shot WBC Classification ===")

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model loaded successfully.")

    # Load test dataset features
    print("Loading test dataset...")
    test_features, test_labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="test"
    )

    # Convert to tensors
    test_features = torch.from_numpy(test_features).float()
    test_labels = torch.from_numpy(test_labels).long()

    print(
        f"Test dataset: {len(test_features)} samples, {len(util.CLASSES)} classes")

    # Load crowded prompts for each class
    print(
        f"\nLoading crowded prompts for {args.few_shot}-shot configuration...")
    binary_probs_dict = {}

    for class_name in util.CLASSES:
        print(f"\nProcessing {class_name}...")

        try:
            # Load crowded prompts for this class
            prompt_list = load_crowded_prompts_for_class(
                args.few_shot, class_name)

            # Evaluate binary ensemble for this class
            binary_probs = evaluate_binary_ensemble(
                prompt_list=prompt_list,
                image_feats=test_features,
                model=model,
                tokenizer=tokenizer,
                class_name=class_name
            )

            binary_probs_dict[class_name] = binary_probs

            print(f"Binary evaluation complete for {class_name}")
            print(f"  Mean probability: {binary_probs.mean():.4f}")
            print(f"  Std probability: {binary_probs.std():.4f}")

        except FileNotFoundError as e:
            print(f"Error loading prompts for {class_name}: {e}")
            print(f"Using zero probabilities for {class_name}")
            binary_probs_dict[class_name] = np.zeros(len(test_features))

    # Convert binary predictions to multiclass
    print(f"\nConverting binary predictions to multiclass...")
    multiclass_probs, multiclass_preds = convert_binary_to_multiclass(
        binary_probs_dict)

    # Evaluate multiclass performance
    print(f"\nEvaluating multiclass performance...")
    y_true = test_labels.numpy()
    y_pred = multiclass_preds

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=util.CLASSES,
        digits=4,
        zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Print results
    print(f"\n=== {args.few_shot}-Shot Multiclass Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(report)

    # Save results
    results_dir = f"final_results/multiclass_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{args.few_shot}-shot-multiclass-results.txt"

    with open(results_file, 'w') as f:
        f.write(
            f"=== {args.few_shot}-Shot Binary-to-Multiclass Evaluation Results ===\n")
        f.write(f"Test samples: {len(test_features)}\n")
        f.write(f"Classes: {util.CLASSES}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n{report}\n")

        # Add per-class binary statistics
        f.write(f"\nPer-Class Binary Statistics:\n")
        for class_name in util.CLASSES:
            if class_name in binary_probs_dict:
                probs = binary_probs_dict[class_name]
                f.write(
                    f"  {class_name}: mean={probs.mean():.4f}, std={probs.std():.4f}\n")

    print(f"\nResults saved to: {results_file}")

    # Optional: Save probability distributions for analysis
    probs_file = f"{results_dir}/{args.few_shot}-shot-multiclass-probabilities.npy"
    np.save(probs_file, multiclass_probs)
    print(f"Probability distributions saved to: {probs_file}")


if __name__ == "__main__":
    main()
