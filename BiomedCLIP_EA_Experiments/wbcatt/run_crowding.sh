#!/bin/bash

# Crowd Pruning Script for All WBC Classes and Few-Shot Configurations
# This script runs the crowding.py script for all combinations of classes and shots

echo "=== Starting Crowd Pruning for All WBC Classes and Few-Shot Configurations ==="
echo "Timestamp: $(date)"

# Define few-shot configurations and WBC classes
FEW_SHOTS=(1 2 4 8)
WBC_CLASSES=("Basophil" "Eosinophil" "Lymphocyte" "Monocyte" "Neutrophil")

# Define other common parameters
PROVIDER="gemini"
MAX_CAPACITY=1000
FILTER_THRESHOLD=0.6
CROWDING_ITERATIONS=3
GROUP_SIZE=30

# Create a logs directory for output
LOGS_DIR="logs/crowding_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"

echo "Logs will be saved to: $LOGS_DIR"

# Function to run crowding for a specific few-shot and class configuration
run_crowding() {
    local few_shot=$1
    local class_label=$2
    local log_file="$LOGS_DIR/${few_shot}shot_${class_label}_crowding.log"
    
    echo "Starting ${few_shot}-shot ${class_label} crowding (PID: $$) at $(date)" > "$log_file"
    
    python crowding.py \
        --few_shot "$few_shot" \
        --class_label "$class_label" \
        --provider "$PROVIDER" \
        --max_capacity "$MAX_CAPACITY" \
        --filter_threshold "$FILTER_THRESHOLD" \
        --crowding_iterations "$CROWDING_ITERATIONS" \
        --group_size "$GROUP_SIZE" \
        >> "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ ${few_shot}-shot ${class_label} crowding completed successfully at $(date)" >> "$log_file"
        echo "✅ ${few_shot}-shot ${class_label} crowding completed successfully"
    else
        echo "❌ ${few_shot}-shot ${class_label} crowding failed with exit code $exit_code at $(date)" >> "$log_file"
        echo "❌ ${few_shot}-shot ${class_label} crowding failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Store process IDs and configurations for monitoring
declare -a PIDS=()
declare -a CONFIGS=()

echo "Starting parallel crowding jobs..."
echo "Total combinations: $((${#FEW_SHOTS[@]} * ${#WBC_CLASSES[@]}))"
echo ""

# Start all crowding jobs in parallel
job_count=0
for few_shot in "${FEW_SHOTS[@]}"; do
    for class_label in "${WBC_CLASSES[@]}"; do
        echo "Starting ${few_shot}-shot ${class_label} crowding..."
        run_crowding "$few_shot" "$class_label" &
        pid=$!
        PIDS+=($pid)
        CONFIGS+=("${few_shot}-shot-${class_label}")
        job_count=$((job_count + 1))
        echo "  └─ Job $job_count: Process started with PID: $pid"
        
        # Optional: Add a small delay to avoid overwhelming the system
        sleep 1
    done
done

echo ""
echo "All $job_count crowding jobs started. Waiting for completion..."
echo "Monitor progress with: tail -f $LOGS_DIR/*_crowding.log"
echo ""

# Wait for all processes to complete and collect results
declare -a RESULTS=()
completed=0
total=${#PIDS[@]}

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    config=${CONFIGS[$i]}
    
    echo "Waiting for $config (PID: $pid)... [$((completed+1))/$total]"
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        RESULTS+=("✅ $config: SUCCESS")
    else
        RESULTS+=("❌ $config: FAILED (exit code: $exit_code)")
    fi
    
    completed=$((completed + 1))
    echo "  └─ $config completed [$completed/$total]"
done

# Create summary report
SUMMARY_FILE="$LOGS_DIR/crowding_summary_report.txt"
echo "=== WBC Crowding Summary Report ===" > "$SUMMARY_FILE"
echo "Execution Date: $(date)" >> "$SUMMARY_FILE"
echo "Script Location: $(pwd)/crowding.py" >> "$SUMMARY_FILE"
echo "Logs Directory: $LOGS_DIR" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Configuration Used:" >> "$SUMMARY_FILE"
echo "  Provider: $PROVIDER" >> "$SUMMARY_FILE"
echo "  Max Capacity: $MAX_CAPACITY" >> "$SUMMARY_FILE"
echo "  Filter Threshold: $FILTER_THRESHOLD" >> "$SUMMARY_FILE"
echo "  Crowding Iterations: $CROWDING_ITERATIONS" >> "$SUMMARY_FILE"
echo "  Group Size: $GROUP_SIZE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Few-Shot Configurations: ${FEW_SHOTS[*]}" >> "$SUMMARY_FILE"
echo "WBC Classes: ${WBC_CLASSES[*]}" >> "$SUMMARY_FILE"
echo "Total Combinations: $total" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "Results:" >> "$SUMMARY_FILE"
for result in "${RESULTS[@]}"; do
    echo "  $result" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "Output Files Generated:" >> "$SUMMARY_FILE"
for few_shot in "${FEW_SHOTS[@]}"; do
    for class_label in "${WBC_CLASSES[@]}"; do
        echo "  ${few_shot}-shot ${class_label}:" >> "$SUMMARY_FILE"
        echo "    - final_results/crowded/${few_shot}-shot-${class_label}.txt" >> "$SUMMARY_FILE"
        echo "    - final_results/crowded/${few_shot}-shot-${class_label}-all.txt" >> "$SUMMARY_FILE"
    done
done

# Print final summary to console
echo ""
echo "=== CROWDING EXECUTION SUMMARY ==="
echo "Completion Time: $(date)"
echo ""
echo "Results:"
for result in "${RESULTS[@]}"; do
    if [[ $result == *"SUCCESS"* ]]; then
        echo "  $result"
    else
        echo "  $result"
    fi
done

echo ""
echo "📊 Summary report saved to: $SUMMARY_FILE"
echo "📁 All logs available in: $LOGS_DIR"

# Check if all crowding jobs were successful
success_count=$(printf '%s\n' "${RESULTS[@]}" | grep -c "SUCCESS")
total_count=${#RESULTS[@]}

if [ $success_count -eq $total_count ]; then
    echo "🎉 All $total_count crowding jobs completed successfully!"
    echo ""
    echo "📋 Output Summary:"
    echo "   - Classes processed: ${WBC_CLASSES[*]}"
    echo "   - Few-shot configs: ${FEW_SHOTS[*]}"
    echo "   - Total files generated: $((total_count * 2)) (top 100 + all crowded)"
    echo "   - Location: final_results/crowded/"
    exit 0
else
    failed_count=$((total_count - success_count))
    echo "⚠️  $failed_count out of $total_count crowding jobs failed."
    echo "   Check individual log files in $LOGS_DIR for details."
    exit 1
fi
