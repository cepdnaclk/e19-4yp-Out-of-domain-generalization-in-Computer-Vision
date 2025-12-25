#!/bin/bash

# Crowding Analysis Runner for Different Experiment Types
# This script demonstrates how to run crowding analysis on different types of experiments,
# including ablation studies with flexible input/output paths

# Exit on any error
set -e

# Function to run crowding analysis
run_crowding_analysis() {
    local input_file=$1
    local output_dir=$2
    local experiment_suffix=$3
    local few_shot=$4
    local provider=${5:-"gemini"}
    
    echo "========================================"
    echo "Running crowding analysis:"
    echo "  Input file: $input_file"
    echo "  Output dir: $output_dir" 
    echo "  Experiment suffix: $experiment_suffix"
    echo "  Few shot: $few_shot"
    echo "  Provider: $provider"
    echo "========================================"
    
    python evaluate_by_crowding_and_knee.py \
        --input_file "$input_file" \
        --output_dir "$output_dir" \
        --experiment_suffix "$experiment_suffix" \
        --few_shot "$few_shot" \
        --provider "$provider"
    
    echo "Crowding analysis completed for: $experiment_suffix"
    echo ""
}

# Example 1: Standard experiment
echo "=== Example 1: Standard Experiment ==="
run_crowding_analysis \
    "final_results/Experiment-70-strategy-inv-bce-gemma3-32shot_opt_pairs.txt" \
    "final_results" \
    "standard" \
    32

# Example 2: Ablation study - different strategy
echo "=== Example 2: Ablation Study - Strategy ==="
run_crowding_analysis \
    "ablation/strategy_ablation/Experiment-71-medical-concepts-inv-bce-gemma3-32shot_opt_pairs.txt" \
    "ablation/strategy_ablation/crowding_results" \
    "medical_concepts_ablation" \
    32

# Example 3: Ablation study - different model
echo "=== Example 3: Ablation Study - Model ==="
run_crowding_analysis \
    "ablation/model_ablation/Experiment-72-strategy-inv-bce-claude3-32shot_opt_pairs.txt" \
    "ablation/model_ablation/crowding_results" \
    "claude3_ablation" \
    32

# Example 4: Ablation study - different metric
echo "=== Example 4: Ablation Study - Metric ==="
run_crowding_analysis \
    "ablation/metric_ablation/Experiment-73-strategy-accuracy-gemma3-32shot_opt_pairs.txt" \
    "ablation/metric_ablation/crowding_results" \
    "accuracy_metric_ablation" \
    32

# Example 5: Few-shot ablation
echo "=== Example 5: Few-shot Ablation ==="
for few_shot in 1 4 8 16 32; do
    run_crowding_analysis \
        "ablation/fewshot_ablation/Experiment-74-strategy-inv-bce-gemma3-${few_shot}shot_opt_pairs.txt" \
        "ablation/fewshot_ablation/crowding_results" \
        "fewshot_${few_shot}_ablation" \
        $few_shot
done

echo "=========================================="
echo "All crowding analyses completed!"
echo "Check the respective output directories for results"
echo "=========================================="
