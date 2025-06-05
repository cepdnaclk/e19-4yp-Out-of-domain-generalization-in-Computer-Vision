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
    negative_prompt = "No evidence of tumor cells with a normal nuclear size, normal nucleolus size, normal nuclear chromatin, normal cell density, normal nuclear border, normal spindle cell morphology, normal nuclear grade"
    positive_prompt = "Large, prominent, and multiple nucleoli with increased nuclear chromatin, high cell density, irregular and jagged nuclear borders, spindle cell morphology, high-grade nuclei with significant atypia."

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
