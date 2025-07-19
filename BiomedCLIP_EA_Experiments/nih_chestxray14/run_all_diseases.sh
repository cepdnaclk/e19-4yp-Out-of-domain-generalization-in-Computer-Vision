#!/bin/bash

# Define diseases list
diseases=(
    "Infiltration"
    "Effusion"
    "Atelectasis"
    "Nodule"
    "Mass"
    "Pneumothorax"
    "Consolidation"
    "Pleural_Thickening"
    "Cardiomegaly"
    "Emphysema"
    "Edema"
    "Fibrosis"
    "Pneumonia"
    "Hernia"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your environment if needed
# source ~/miniconda3/bin/activate myenv

# Loop through each disease and run optimization
for disease in "${diseases[@]}"
do
    echo "==============================="
    echo "Running optimization for $disease"
    echo "==============================="

    # Run with nohup, but wait for completion before starting next
    nohup python prompt_optimizer.py "$disease" > logs/"$disease"_output.log 2>&1

    # Wait until the process finishes before starting next
    wait

    echo "Completed optimization for $disease"
done

echo "All diseases processed sequentially."
