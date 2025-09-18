#!/bin/bash

# WBCATT Binary Classification Experiment Runner
# This script runs prompt optimization experiments for all combinations of:
# - Few-shot sizes: 1, 2, 4, 8
# - Classes: 0 (Basophil), 1 (Eosinophil), 2 (Lymphocyte), 3 (Monocyte), 4 (Neutrophil)
# - Fitness metric: inverted_weighted_ce (can be modified)

# Exit on any error
set -e

# Configuration
FITNESS_METRIC="inverted_weighted_ce"
FEW_SHOTS=(1 2 4 8)
CLASSES=(0 1 2 3 4)
CLASS_NAMES=("Basophil" "Eosinophil" "Lymphocyte" "Monocyte" "Neutrophil")

# Create logs directory
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local class_idx=$1
    local few_shot=$2
    local class_name=${CLASS_NAMES[$class_idx]}
    
    echo "========================================"
    echo "Running experiment:"
    echo "  Class: $class_name (index: $class_idx)"
    echo "  Few-shot: $few_shot"
    echo "  Fitness metric: $FITNESS_METRIC"
    echo "========================================"
    
    # Create log filename
    log_file="logs/experiment_class${class_idx}_${class_name}_fewshot${few_shot}_${FITNESS_METRIC}.log"
    
    # Run the experiment
    python prompt_optimizer_binary.py \
        --binary_label $class_idx \
        --fitness_metric $FITNESS_METRIC \
        --few_shot $few_shot \
        2>&1 | tee "$log_file"
    
    echo "Experiment completed. Log saved to: $log_file"
    echo ""
}

# Function to run all few-shot experiments for a single class in parallel
run_class_experiments() {
    local class_idx=$1
    local class_name=${CLASS_NAMES[$class_idx]}
    
    echo "Starting all few-shot experiments for class: $class_name (index: $class_idx)"
    
    # Run all few-shot sizes for this class in parallel
    for few_shot in "${FEW_SHOTS[@]}"; do
        run_experiment $class_idx $few_shot &
    done
    
    # Wait for all few-shot experiments for this class to complete
    wait
    echo "All few-shot experiments completed for class: $class_name"
    echo ""
}

# Print experiment plan
echo "=========================================="
echo "WBCATT Binary Classification Experiments"
echo "=========================================="
echo "Total experiments planned: $((${#FEW_SHOTS[@]} * ${#CLASSES[@]}))"
echo "Few-shot sizes: ${FEW_SHOTS[*]}"
echo "Classes: ${CLASS_NAMES[*]}"
echo "Fitness metric: $FITNESS_METRIC"
echo "Parallelization: Each class runs all few-shot sizes in parallel"
echo ""
echo "Starting experiments..."
echo ""

# Run experiments for each class (classes run sequentially, but few-shot sizes within each class run in parallel)
for class_idx in "${CLASSES[@]}"; do
    run_class_experiments $class_idx
done

echo "=========================================="
echo "All experiments completed!"
echo "Check the 'logs/' directory for individual experiment logs"
echo "Check the 'final_results/' directory for optimization results"
echo "=========================================="
