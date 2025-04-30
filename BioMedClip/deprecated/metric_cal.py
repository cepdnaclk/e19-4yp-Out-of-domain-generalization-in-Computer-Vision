import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(csv_path):
    df = pd.read_csv(csv_path)
    
    df = df[df['ground_truth'] != -1]
    
    if len(df) == 0:
        print("No samples with valid ground truth found!")
        return
    
    y_true = df['ground_truth']
    
    y_pred = (df['positive_score'] > df['negative_score']).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    y_pred_prob = df['positive_score']
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix (Positive > Negative)')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='negative_score', y='positive_score', hue='ground_truth')
    plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line where positive = negative
    plt.title('Positive vs Negative Scores by Ground Truth')
    plt.xlabel('Negative Score (Tumor not present)')
    plt.ylabel('Positive Score (Tumor present)')
    plt.show()
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    results_csv = "biomedclip_results.csv"
    
    metrics = calculate_metrics(results_csv)
    
    if metrics:
        pd.DataFrame.from_dict(metrics, orient='index').to_csv("evaluation_metrics.csv")
        print("\nMetrics saved to evaluation_metrics.csv")