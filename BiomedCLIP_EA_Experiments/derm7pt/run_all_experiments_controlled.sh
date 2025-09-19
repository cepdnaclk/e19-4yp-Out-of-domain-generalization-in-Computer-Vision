#!/bin/bash

# Derm7pt Prompt Optimization Experiment Runner - Controlled Parallel Version
# This script runs prompt optimization experiments for all combinations with controlled parallelization:
# - Few-shot sizes: 0, 1, 2, 4, 8, 16
# - Fitness metrics: inverted_ce, f1_macro
# Metrics run sequentially, but all few-shot sizes within each metric run in parallel

# Exit on any error
set -e

# Configuration
FEW_SHOTS=(0 1 2 4 8 16)
FITNESS_METRICS=("inverted_ce" "f1_macro")

# Create logs directory
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local few_shot=$1
    local fitness_metric=$2
    
    echo "========================================"
    echo "Running experiment:"
    echo "  Few-shot: $few_shot"
    echo "  Fitness metric: $fitness_metric"
    echo "========================================"
    
    # Create log filename
    log_file="logs/experiment_fewshot${few_shot}_${fitness_metric}.log"
    
    # Run the experiment
    python prompt_optimizer.py \
        --fitness_metric $fitness_metric \
        --few_shot $few_shot \
        2>&1 | tee "$log_file"
    
    echo "Experiment completed. Log saved to: $log_file"
    echo ""
}

# Function to run all few-shot experiments for a single metric in parallel
run_metric_experiments() {
    local fitness_metric=$1
    
    echo "Starting all few-shot experiments for metric: $fitness_metric"
    
    # Run all few-shot sizes for this metric in parallel
    for few_shot in "${FEW_SHOTS[@]}"; do
        run_experiment $few_shot $fitness_metric &
    done
    
    # Wait for all few-shot experiments for this metric to complete
    wait
    echo "All few-shot experiments completed for metric: $fitness_metric"
    echo ""
}

# Print experiment plan
echo "=========================================="
echo "Derm7pt Prompt Optimization Experiments"
echo "=========================================="
echo "Total experiments planned: $((${#FEW_SHOTS[@]} * ${#FITNESS_METRICS[@]}))"
echo "Few-shot sizes: ${FEW_SHOTS[*]}"
echo "Fitness metrics: ${FITNESS_METRICS[*]}"
echo "Parallelization: Each metric runs all few-shot sizes in parallel"
echo ""
echo "Starting experiments..."
echo ""

# Run experiments for each metric (metrics run sequentially, but few-shot sizes within each metric run in parallel)
for fitness_metric in "${FITNESS_METRICS[@]}"; do
    run_metric_experiments $fitness_metric
done

echo "=========================================="
echo "All experiments completed!"
echo "Check the 'logs/' directory for individual experiment logs"
echo "Check the 'final_results/' directory for optimization results"
echo "=========================================="
