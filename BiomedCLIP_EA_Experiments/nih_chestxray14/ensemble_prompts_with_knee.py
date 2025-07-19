from typing import List, Optional
import util
import torch
import numpy as np
import os


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset
    # 1) Unpack—annotate what extract_center_embeddings returns
    centers_features: List[np.ndarray]
    centers_labels:   List[np.ndarray]
    

    features, labels = util.extract_embeddings(
        model=model,
        preprocess=preprocess,
        train_or_test="test"  # Evaluating on test centers only
    )

    print("embeddings extracted successfully.")
    # Convert to torch tensors for each center
    # Convert to torch tensors for each center
    all_features = torch.from_numpy(features)
    all_labels = torch.from_numpy(labels)
    # 3. load prompts
    initial_prompts = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt")

    print(f"Initial prompts loaded: {len(initial_prompts)} prompts.")

    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)

    # knee point analysis
    all_current_scores = [score for _, score in pq.get_best_n(
        len(pq))]  # Get all, sorted by score
    knee_analyzer = util.KneePointAnalysis(all_current_scores)
    recommended_n = knee_analyzer.find_knee_point()

    print(
        f"Recommended number of prompts after knee analysis: {recommended_n}")

    prompts_population = pq.get_best_n(recommended_n)

    for i in range(0, len(prompts_population), 1):
        prompts = prompts_population[0:i + 1]
        print(f"Using {len(prompts)} prompts for evaluation.")

        # Run the evaluation
        for i, _ in enumerate(centers_features):
            print(f"Evaluating center {i}...")
            results = util.evaluate_prompt_list(
                prompts,
                all_features,
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
