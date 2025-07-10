from typing import List
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score
import util


def train_meta_models(P_train, y_train, max_depth=3, C=1.0):
    """
    Fit a logistic regressor and a small decision tree.
    Returns both models.
    """
    logreg = LogisticRegression(
        penalty='l2', C=C, solver='liblinear', max_iter=100)
    logreg.fit(P_train, y_train)

    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(P_train, y_train)

    return logreg, tree


def main():
    # 1. load model, preprocess, and tokenizer
    model, preprocess, tokenizer = util.load_clip_model()
    print("Model, preprocess, and tokenizer loaded successfully.")

    # 2. load dataset embeddings by center
    # convert to torch and numpy
    centers_features: List[np.ndarray]
    centers_labels:   List[np.ndarray]
    centers_features, centers_labels = util.extract_center_embeddings(
        model=model,
        preprocess=preprocess,
        num_centers=5,  # evaluating on all centers
        isTrain=False,  # Evaluating on test centers only
    )

    # convert to torch tensors for each center
    centers_features = [torch.from_numpy(feat) for feat in centers_features]
    centers_labels = [torch.from_numpy(label) for label in centers_labels]
    print("Center embeddings extracted successfully.")

    # 3. load and select prompts
    initial_prompts = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt"
    )
    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)
    all_scores = [s for _, s in pq.get_best_n(len(pq))]
    recommended_n = util.KneePointAnalysis(all_scores).find_knee_point()
    prompt_population = [p for p, _ in pq.get_best_n(recommended_n)]
    print(f"Selected {len(prompt_population)} prompts after knee analysis.")

    # 4. build train/test splits by center indices
    train_idxs = [0, 1, 2]
    test_idxs = [3, 4]

    P_train_list, y_train_list = [], []
    P_test_list, y_test_list = [], []

    for idx in train_idxs:
        feats = centers_features[idx]
        labels = centers_labels[idx]
        P = util.compute_prompt_probs_matrix(
            prompt_population, feats, model, tokenizer)
        P_train_list.append(P)
        y_train_list.append(labels)

    for idx in test_idxs:
        feats = centers_features[idx]
        labels = centers_labels[idx]
        P = util.compute_prompt_probs_matrix(
            prompt_population, feats, model, tokenizer)
        P_test_list.append(P)
        y_test_list.append(labels)

    # stack centers
    P_train = np.vstack(P_train_list)
    y_train = np.concatenate(y_train_list)
    P_test = np.vstack(P_test_list)
    y_test = np.concatenate(y_test_list)

    print(
        f"Training meta‑models on centers {train_idxs}: {P_train.shape[0]} samples")
    print(f"Evaluating on centers {test_idxs}: {P_test.shape[0]} samples")

    # 5. train and evaluate
    logreg, tree = train_meta_models(P_train, y_train)

    # test metrics
    y_prob = logreg.predict_proba(P_test)[:, 1]
    y_pred = logreg.predict(P_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest Meta‑LogReg  →  Acc: {acc:.4f}, AUC: {auc:.4f}")
    print("Prompt coefficients:")
    for j, coef in enumerate(logreg.coef_[0]):
        print(f"  Prompt #{j:2d}:  β = {coef:.4f}")

    print("\nDecision‑Tree rules:\n")
    print(export_text(tree, feature_names=[
          f"p{j}" for j in range(P_train.shape[1])]))


if __name__ == '__main__':
    main()
