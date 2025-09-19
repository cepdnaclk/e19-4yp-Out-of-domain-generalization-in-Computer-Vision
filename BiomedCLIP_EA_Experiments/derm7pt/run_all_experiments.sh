#!/bin/bash

# Derm7pt Prompt Optimization Experiment Runner
# This script runs prompt optimization experiments for all combinations of:
# - Few-shot sizes: 0, 1, 2, 4, 8, 16
# - Fitness metrics: inverted_ce, f1_macro
# All experiments run in parallel for maximum speed

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

# Print experiment plan
echo "=========================================="
echo "Derm7pt Prompt Optimization Experiments"
echo "=========================================="
echo "Total experiments planned: $((${#FEW_SHOTS[@]} * ${#FITNESS_METRICS[@]}))"
echo "Few-shot sizes: ${FEW_SHOTS[*]}"
echo "Fitness metrics: ${FITNESS_METRICS[*]}"
echo "Parallelization: ALL experiments run in parallel"
echo ""
echo "Starting all experiments in parallel..."
echo ""

# Start all experiments in parallel
for few_shot in "${FEW_SHOTS[@]}"; do
    for fitness_metric in "${FITNESS_METRICS[@]}"; do
        run_experiment $few_shot $fitness_metric &
    done
done

# Wait for all experiments to complete
echo "Waiting for all experiments to complete..."
wait

echo "=========================================="
echo "All experiments completed!"
echo "Check the 'logs/' directory for individual experiment logs"
echo "Check the 'final_results/' directory for optimization results"
echo "=========================================="
