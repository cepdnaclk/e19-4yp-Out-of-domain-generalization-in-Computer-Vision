from typing import List
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
    prompts = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt"
    )

    for ((negative_prompt, positive_prompt), score) in prompts:
        print(f"Evaluating prompts: \n")
        print(f"Negative Prompt: {negative_prompt}")
        print(f"Positive Prompt: {positive_prompt}")
        print(f"Training Fitness: {score}\n")
        for i, _ in enumerate(centers_features):
            print(f"Evaluating center {i}...")
            results = util.evaluate_prompt_pair(
                negative_prompt, positive_prompt, centers_features[i], centers_labels[i], model, tokenizer)

            print(f"Accuracy: {results['accuracy']}")
            print(f"ROC AUC: {results['auc']}")
            print(f"Confusion Matrix:\n{results['cm']}")
            print(f"Classification Report:\n{results['report']}")


if __name__ == "__main__":
    main()
