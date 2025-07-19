from typing import List
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import util


def train_meta_models(P_train, y_train, max_depth=5, C=1.0):
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

    # use CPU
    util.DEVICE = torch.device("cpu")

    # 3. load and select prompts
    initial_prompts = util.load_initial_prompts(
        "experiment_results/distinct_medical_concepts.txt"
    )
    pq = util.PriorityQueue(
        max_capacity=1000, filter_threshold=0.6, initial=initial_prompts)
    all_scores = [s for _, s in pq.get_best_n(len(pq))]
    recommended_n = util.KneePointAnalysis(all_scores).find_knee_point()
    prompt_population = pq.get_best_n(recommended_n)
    print(f"Selected {len(prompt_population)} prompts after knee analysis.")

    # Extract scores for normalization
    prompt_scores = np.array([score for _, score in prompt_population])

    # Initialize and fit standard scaler
    scaler = StandardScaler()
    scores_standardized = scaler.fit_transform(
        prompt_scores.reshape(-1, 1)).flatten()

    print(
        f"Original scores range: [{prompt_scores.min():.4f}, {prompt_scores.max():.4f}]")
    print(
        f"Standardized scores range: [{scores_standardized.min():.4f}, {scores_standardized.max():.4f}]")

    # 4. build train/test splits by center indices
    train_idxs = [0, 1, 2]
    test_idxs = [4]

    P_train_list, y_train_list = [], []
    P_test_list, y_test_list = [], []

    for idx in train_idxs:
        feats = centers_features[idx]
        labels = centers_labels[idx]
        P = util.compute_prompt_probs_matrix(
            prompt_population, feats, model, tokenizer)

        # Add standardized scores as additional features
        n_samples = P.shape[0]
        scores_repeated = np.tile(scores_standardized, (n_samples, 1))

        # Concatenate probability matrix with standardized scores
        P_with_scores = np.hstack([P, scores_repeated])

        P_train_list.append(P_with_scores)
        y_train_list.append(labels)

    for idx in test_idxs:
        feats = centers_features[idx]
        labels = centers_labels[idx]
        P = util.compute_prompt_probs_matrix(
            prompt_population, feats, model, tokenizer)

        # Add standardized scores as additional features
        n_samples = P.shape[0]
        scores_repeated = np.tile(scores_standardized, (n_samples, 1))

        # Concatenate probability matrix with standardized scores
        P_with_scores = np.hstack([P, scores_repeated])

        P_test_list.append(P_with_scores)
        y_test_list.append(labels)

    # stack centers
    P_train = np.vstack(P_train_list)
    y_train = np.concatenate(y_train_list)
    P_test = np.vstack(P_test_list)
    y_test = np.concatenate(y_test_list)

    n_prompts = len(prompt_population)
    print(
        f"Training meta‑models on centers {train_idxs}: {P_train.shape[0]} samples")
    print(
        f"Feature shape: {P_train.shape[1]} ({n_prompts} prompt probs + {n_prompts} standardized scores)")
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
        if j < n_prompts:
            print(f"  Prompt #{j:2d} (prob):  β = {coef:.4f}")
        else:
            print(f"  Prompt #{j-n_prompts:2d} (score): β = {coef:.4f}")

    # Decision tree evaluation
    y_tree_pred = tree.predict(P_test)
    tree_acc = accuracy_score(y_test, y_tree_pred)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(P_test)[:, 1])
    print(
        f"\nTest Meta‑DecisionTree  →  Acc: {tree_acc:.4f}, AUC: {tree_auc:.4f}")

    print("\nDecision‑Tree rules:\n")
    feature_names = []
    for j in range(n_prompts):
        feature_names.append(f"p{j}_prob")
    for j in range(n_prompts):
        feature_names.append(f"p{j}_score")

    print(export_text(tree, feature_names=feature_names))


if __name__ == '__main__':
    main()
