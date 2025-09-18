#!/bin/bash

# Crowd Pruning and Knee Point Analysis - All Few-Shot Configurations
# This script runs the evaluation for 1-shot, 2-shot, 4-shot, and 8-shot configurations in parallel

echo "=== Starting Crowd Pruning and Knee Point Analysis for All Few-Shot Configurations ==="
echo "Timestamp: $(date)"

# Define few-shot configurations to run
FEW_SHOTS=(1 2 4 8)

# Define other common parameters
PROVIDER="gemini"
MAX_CAPACITY=1000
FILTER_THRESHOLD=0.6
CROWDING_ITERATIONS=3
GROUP_SIZE=30

# Create a logs directory for output
LOGS_DIR="logs/crowd_pruning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"

echo "Logs will be saved to: $LOGS_DIR"

# Function to run evaluation for a specific few-shot configuration
run_evaluation() {
    local few_shot=$1
    local log_file="$LOGS_DIR/${few_shot}shot_evaluation.log"
    
    echo "Starting ${few_shot}-shot evaluation (PID: $$) at $(date)" > "$log_file"
    
    python evaluate_by_crowding_and_knee.py \
        --few_shot "$few_shot" \
        --provider "$PROVIDER" \
        --max_capacity "$MAX_CAPACITY" \
        --filter_threshold "$FILTER_THRESHOLD" \
        --crowding_iterations "$CROWDING_ITERATIONS" \
        --group_size "$GROUP_SIZE" \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${few_shot}-shot evaluation completed successfully at $(date)" >> "$log_file"
        echo "✅ ${few_shot}-shot evaluation completed successfully"
    else
        echo "❌ ${few_shot}-shot evaluation failed with exit code $exit_code at $(date)" >> "$log_file"
        echo "❌ ${few_shot}-shot evaluation failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Store process IDs for monitoring
declare -a PIDS=()
declare -a CONFIGS=()

echo "Starting parallel evaluations..."

# Start all evaluations in parallel
for few_shot in "${FEW_SHOTS[@]}"; do
    echo "Starting ${few_shot}-shot evaluation..."
    run_evaluation "$few_shot" &
    pid=$!
    PIDS+=($pid)
    CONFIGS+=($few_shot)
    echo "  └─ Process started with PID: $pid"
done

echo ""
echo "All evaluations started. Waiting for completion..."
echo "Monitor progress with: tail -f $LOGS_DIR/*shot_evaluation.log"
echo ""

# Wait for all processes to complete and collect results
declare -a RESULTS=()
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    config=${CONFIGS[$i]}
    
    echo "Waiting for ${config}-shot evaluation (PID: $pid)..."
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        RESULTS+=("✅ ${config}-shot: SUCCESS")
    else
        RESULTS+=("❌ ${config}-shot: FAILED (exit code: $exit_code)")
    fi
done

# Create summary report
SUMMARY_FILE="$LOGS_DIR/summary_report.txt"
echo "=== Crowd Pruning and Knee Point Analysis Summary ===" > "$SUMMARY_FILE"
echo "Execution Date: $(date)" >> "$SUMMARY_FILE"
echo "Script Location: $(pwd)/evaluate_by_crowding_and_knee.py" >> "$SUMMARY_FILE"
echo "Logs Directory: $LOGS_DIR" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Configuration Used:" >> "$SUMMARY_FILE"
echo "  Provider: $PROVIDER" >> "$SUMMARY_FILE"
echo "  Max Capacity: $MAX_CAPACITY" >> "$SUMMARY_FILE"
echo "  Filter Threshold: $FILTER_THRESHOLD" >> "$SUMMARY_FILE"
echo "  Crowding Iterations: $CROWDING_ITERATIONS" >> "$SUMMARY_FILE"
echo "  Group Size: $GROUP_SIZE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Results:" >> "$SUMMARY_FILE"
for result in "${RESULTS[@]}"; do
    echo "  $result" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "Output Files Generated:" >> "$SUMMARY_FILE"
for few_shot in "${FEW_SHOTS[@]}"; do
    echo "  ${few_shot}-shot results:" >> "$SUMMARY_FILE"
    echo "    - final_results/crowded/${few_shot}-shot.txt" >> "$SUMMARY_FILE"
    echo "    - final_results/knee/${few_shot}-shot.txt" >> "$SUMMARY_FILE"
    echo "    - final_results/evaluation/${few_shot}-shot-results.txt" >> "$SUMMARY_FILE"
done

# Print final summary to console
echo ""
echo "=== EXECUTION SUMMARY ==="
echo "Completion Time: $(date)"
echo ""
echo "Results:"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo "📊 Summary report saved to: $SUMMARY_FILE"
echo "📁 All logs available in: $LOGS_DIR"

# Check if all evaluations were successful
success_count=$(printf '%s\n' "${RESULTS[@]}" | grep -c "SUCCESS")
total_count=${#RESULTS[@]}

if [ $success_count -eq $total_count ]; then
    echo "🎉 All evaluations completed successfully!"
    exit 0
else
    echo "⚠️  $((total_count - success_count)) out of $total_count evaluations failed."
    echo "   Check individual log files for details."
    exit 1
fi
