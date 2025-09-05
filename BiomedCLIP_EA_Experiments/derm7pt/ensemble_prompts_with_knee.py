from typing import List, Optional
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
    init_prompts = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt"
    )

    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.3, initial=init_prompts)

    # knee point analysis
    all_current_scores = [score for _, score in pq.get_best_n(
        len(pq))]  # Get all, sorted by score
    knee_analyzer = util.KneePointAnalysis(all_current_scores)
    recommended_n = knee_analyzer.find_knee_point()

    print(
        f"Recommended number of prompts after knee analysis: {recommended_n}")

    prompts_population = pq.get_best_n(recommended_n)

    results = util.evaluate_prompt_list(
        prompts_population,
        all_feats,
        all_labels,
        model,
        tokenizer,
        unweighted=False
    )

    print("\n--- Ensemble Evaluation Results ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print("Confusion Matrix:\n", results['cm'])
    print("Classification Report:\n", results['report'])


if __name__ == "__main__":
    main()
