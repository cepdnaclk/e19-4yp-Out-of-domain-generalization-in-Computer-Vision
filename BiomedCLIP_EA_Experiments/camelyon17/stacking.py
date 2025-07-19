from typing import List
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import util


def train_meta_models(P_train, y_train, max_depth=3, C=1.0):
    """
    Fit multiple meta models and return them.
    Returns a dictionary of trained models.
    """
    models = {}

    # Logistic Regression
    models['logistic'] = LogisticRegression(
        penalty='l2', C=C, solver='liblinear', max_iter=1000)
    models['logistic'].fit(P_train, y_train)

    # Decision Tree
    models['tree'] = DecisionTreeClassifier(max_depth=max_depth)
    models['tree'].fit(P_train, y_train)

    # Random Forest
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100, max_depth=max_depth+2, random_state=42, n_jobs=-1)
    models['random_forest'].fit(P_train, y_train)

    # Gradient Boosting
    models['gradient_boosting'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=max_depth, random_state=42)
    models['gradient_boosting'].fit(P_train, y_train)

    # Multi-layer Perceptron
    models['mlp'] = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42,
        alpha=0.01, early_stopping=True, validation_fraction=0.1)
    models['mlp'].fit(P_train, y_train)

    # Support Vector Machine (with probability estimates)
    models['svm'] = SVC(probability=True, random_state=42, gamma='scale')
    models['svm'].fit(P_train, y_train)

    # Gaussian Naive Bayes
    models['naive_bayes'] = GaussianNB()
    models['naive_bayes'].fit(P_train, y_train)

    return models


def evaluate_meta_models(models, P_val, y_val):
    """
    Evaluate multiple meta models and return a comparison summary.
    """
    print("\n" + "="*60)
    print("META-MODEL PERFORMANCE COMPARISON")
    print("="*60)

    results = {}

    for name, model in models.items():
        # Get positive class probabilities
        y_pred_proba = model.predict_proba(P_val)[:, 1]
        y_pred = model.predict(P_val)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_val)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"\n{name.upper().replace('_', ' ')} RESULTS:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

    # Find best model by AUC
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])

    print("\n" + "-"*40)
    print("SUMMARY:")
    print(
        f"Best model by AUC: {best_model_name.upper().replace('_', ' ')} (AUC: {results[best_model_name]['auc']:.4f})")

    # Sort by AUC for ranking
    sorted_models = sorted(
        results.items(), key=lambda x: x[1]['auc'], reverse=True)
    print("\nRanking by AUC:")
    for i, (name, metrics) in enumerate(sorted_models, 1):
        print(f"  {i}. {name.replace('_', ' ').title()}: {metrics['auc']:.4f}")

    print("="*60)

    return results, best_model_name


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
    prompt_population = pq.get_best_n(recommended_n)
    print(f"Selected {len(prompt_population)} prompts after knee analysis.")

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

    n_prompts = len(prompt_population)
    print(
        f"Training meta‑models on centers {train_idxs}: {P_train.shape[0]} samples")
    print(
        f"Feature shape: {P_train.shape[1]} (prompt probabilities only)")
    print(f"Evaluating on centers {test_idxs}: {P_test.shape[0]} samples")

    # 5. train and evaluate multiple meta models
    models = train_meta_models(P_train, y_train)

    # Create a subset of test data for validation during evaluation
    P_val = P_test
    y_val = y_test

    # Evaluate all models
    results, best_model_name = evaluate_meta_models(models, P_val, y_val)

    # Display detailed results for best model
    best_model = models[best_model_name]

    print(f"\n{'='*60}")
    print(
        f"DETAILED ANALYSIS FOR BEST MODEL: {best_model_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    if hasattr(best_model, 'coef_'):  # Linear models have coefficients
        print("Feature coefficients:")
        for j, coef in enumerate(best_model.coef_[0]):
            print(f"  Prompt #{j:2d} (prob):  β = {coef:.4f}")

    if hasattr(best_model, 'feature_importances_'):  # Tree-based models
        print("Feature importances:")
        importances = best_model.feature_importances_
        for j, importance in enumerate(importances):
            print(f"  Prompt #{j:2d} (prob):  importance = {importance:.4f}")

    # Show decision tree rules if the best model is a tree-based model
    if best_model_name in ['tree', 'random_forest'] and hasattr(best_model, 'estimators_'):
        print("\nRandom Forest - First Tree Rules:")
        feature_names = []
        for j in range(n_prompts):
            feature_names.append(f"p{j}_prob")
        print(export_text(best_model.estimators_[
              0], feature_names=feature_names))
    elif best_model_name == 'tree':
        print("\nDecision Tree Rules:")
        feature_names = []
        for j in range(n_prompts):
            feature_names.append(f"p{j}_prob")
        print(export_text(best_model, feature_names=feature_names))


if __name__ == '__main__':
    main()
