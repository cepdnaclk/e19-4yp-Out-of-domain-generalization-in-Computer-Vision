from typing import List, Optional
import util
import torch
import numpy as np
import os
import crowding


def main():
    # 1. load model, process, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset
    # 1) Unpack—annotate what extract_center_embeddings returns
    centers_features: List[np.ndarray]
    centers_labels:   List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=5,  # evaluating on all centers
        isTrain=False,  # Evaluating on test centers only
    )
    print("Center embeddings extracted successfully.")

    # Convert to torch tensors for each center
    centers_features = [torch.from_numpy(feat) for feat in centers_features]
    centers_labels = [torch.from_numpy(label) for label in centers_labels]

    # 3. load prompts
    initial_prompts = util.load_last_iteration_prompts(
        "experiment_results/Experiment-40-strategy_bce_inverted-gemma3_opt_pairs.txt")

    print(f"Initial prompts loaded: {len(initial_prompts)} prompts.")

    if not initial_prompts:
        print("No initial prompts found. Exiting.")
        return

    # 4. Perform crowding pruning
    # pq = util.PriorityQueue(
    #     max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)
    pq = crowding.perform_crowding_pruning(
        initial_prompts=initial_prompts,
        number_of_prompts_to_group=30,
        crowding_iterations=10,
        max_retries=5
    )

    # save crowded prompts
    crowded_prompts_filename = "experiment_results/Experiment-40-crowded_prompts.txt"
    with open(crowded_prompts_filename, 'w') as f:
        for (neg_prompt, pos_prompt), score in pq.get_best_n(len(pq)):
            f.write(f"('{neg_prompt}', '{pos_prompt}'), Score: {score}\n")

    # knee point analysis
    all_current_scores = [score for _, score in pq.get_best_n(
        len(pq))]  # Get all, sorted by score
    knee_analyzer = util.KneePointAnalysis(all_current_scores)
    recommended_n = knee_analyzer.find_knee_point()

    print(
        f"Recommended number of prompts after knee analysis: {recommended_n}")

    prompts_population = pq.get_best_n(recommended_n)
    print(f"Using {len(prompts_population)} prompts for evaluation.")

    # Run the evaluation
    for i, _ in enumerate(centers_features):
        print(f"Evaluating center {i}...")
        results = util.evaluate_prompt_list(
            prompts_population,
            centers_features[i],
            centers_labels[i],
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
