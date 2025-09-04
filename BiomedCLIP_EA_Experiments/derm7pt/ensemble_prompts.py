from typing import List
import util
import torch
import numpy as np
import os


def main():
    label_type = "melanoma"

    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset - MODIFIED FOR CHEXPERT
    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        split="test",
        label_type=label_type,
    )

    # Convert to tensors - MODIFIED FOR MULTI-OBSERVATION SUPPORT
    all_feats = torch.from_numpy(features).float()
    all_labels = torch.from_numpy(labels).long()

    # 3. load prompts
    prompts_population = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt"
    )

    print(f"Using {len(prompts_population)} prompts_population for evaluation.")

    # Run the evaluation
    # for i, _ in enumerate(all_features):
    # print(f"Evaluating center {i}...")
    results = util.evaluate_prompt_list(
        prompts_population,
        all_feats,
        all_labels,
        model,
        tokenizer,
        unweighted=True
    )

    print("\n--- Ensemble Evaluation Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("Confusion Matrix:\n", results['cm'])
    print("Classification Report:\n", results['report'])


if __name__ == "__main__":
    main()
