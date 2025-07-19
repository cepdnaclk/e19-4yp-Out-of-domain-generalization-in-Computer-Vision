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
    
    # Run with nohup and log output per disease
    nohup python prompt_optimizer.py "$disease" > logs/"$disease"_output.log 2>&1

    echo "Started optimization for $disease with nohup. Check logs/$disease_output.log for progress."
done

echo "All diseases submitted (running sequentially under nohup)."
