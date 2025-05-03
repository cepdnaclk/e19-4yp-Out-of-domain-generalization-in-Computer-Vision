import pandas as pd
import os
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

def calculate_metrics_by_center(results_dir, output_dir="center_metrics"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionary to store all metrics
    all_metrics = {}
    
    # Get all center result files
    center_files = [f for f in os.listdir(results_dir) if f.startswith('center_') and f.endswith('.csv')]
    
    if not center_files:
        print("No center result files found!")
        return
    
    # Process each center file
    for center_file in center_files:
        center_num = center_file.split('_')[1].split('.')[0]
        try:
            center_num = int(center_num)
        except:
            continue
        
        file_path = os.path.join(results_dir, center_file)
        df = pd.read_csv(file_path)
        
        # Filter out samples without ground truth
        df = df[df['ground_truth'] != -1]
        
        if len(df) == 0:
            print(f"Center {center_num}: No samples with valid ground truth")
            continue
        
        # Get ground truth and predictions
        y_true = df['ground_truth']
        y_pred = (df['positive_score'] > df['negative_score']).astype(int)
        y_pred_prob = df['positive_score']
        
        # Calculate metrics
        metrics = {
            'center': center_num,
            'num_samples': len(df),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_prob),
            'pr_auc': average_precision_score(y_true, y_pred_prob)
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Store metrics
        all_metrics[center_num] = metrics
        
        # Print center results
        print(f"\nCenter {center_num} Metrics (n={len(df)}):")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Save individual center plots
        plot_center_metrics(metrics, output_dir, center_num)
    
    # Save all metrics to CSV
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_csv = os.path.join(output_dir, "all_center_metrics.csv")
    metrics_df.to_csv(metrics_csv)
    print(f"\nAll center metrics saved to {metrics_csv}")
    
    # Plot comparison across centers
    plot_center_comparison(all_metrics, output_dir)
    
    return all_metrics

def plot_center_metrics(metrics, output_dir, center_num):
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Center {center_num} Confusion Matrix (n={metrics["num_samples"]})')
    plt.savefig(os.path.join(plots_dir, f'center_{center_num}_confusion_matrix.png'))
    plt.close()
    
    # Score distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(metrics.get('positive_scores', []), bins=20, alpha=0.5, label='Positive Scores')
    plt.hist(metrics.get('negative_scores', []), bins=20, alpha=0.5, label='Negative Scores')
    plt.title(f'Center {center_num} Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'center_{center_num}_score_distribution.png'))
    plt.close()

def plot_center_comparison(all_metrics, output_dir):
    if not all_metrics:
        return
    
    # Prepare data for comparison
    centers = []
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    metric_values = {name: [] for name in metrics_names}
    sample_counts = []
    
    for center_num, metrics in sorted(all_metrics.items()):
        centers.append(f"Center {center_num}")
        sample_counts.append(metrics['num_samples'])
        for name in metrics_names:
            metric_values[name].append(metrics[name])
    
    # Plot metric comparison
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(metrics_names):
        plt.plot(centers, metric_values[name], marker='o', label=name)
    
    plt.title('Metric Comparison Across Centers')
    plt.xlabel('Center')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Add sample size annotations
    for i, count in enumerate(sample_counts):
        plt.annotate(f'n={count}', (i, 0), xytext=(0, 10), 
                     textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_metric_comparison.png'))
    plt.close()

if __name__ == "__main__":
    # Path to the directory containing center result files
    results_dir = "biomedclip_results_by_center"
    
    # Calculate and display metrics for each center
    all_metrics = calculate_metrics_by_center(results_dir)