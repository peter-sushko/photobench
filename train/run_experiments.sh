#!/bin/bash

# Directory containing your config files
CONFIG_DIR="/mmfs1/gscratch/krishna/psushko/photobench/finetuning2/instruct-pix2pix/configs/hyperparam_sweep"

# Base command
BASE_CMD="python main_save_every_5_epochs.py"

# Iterate over each config file in the directory
for config_path in "$CONFIG_DIR"/*.yaml; do
    # Extract the config filename
    config_filename=$(basename "$config_path")

    # Skip the specific config that has already been run
    if [ "$config_filename" == "config_lr0.0001_bs16.yaml" ]; then
        echo "Skipping already run config: $config_filename"
        continue
    fi

    if [ "$config_filename" == "config_lr0.0001_bs32.yaml" ]; then
        echo "Skipping already run config: $config_filename"
        continue
    fi

    if [ "$config_filename" == "config_lr0.0001_bs64.yaml" ]; then
        echo "Skipping already run config: $config_filename"
        continue
    fi

    # Extract hyperparameter values from the filename
    # Assuming filenames are like: config_lr1e-07_bs16.yaml
    lr=$(echo "$config_filename" | sed -n 's/.*_lr\([^-]*\)_bs.*/\1/p')
    bs=$(echo "$config_filename" | sed -n 's/.*_bs\([^.]*\).yaml/\1/p')

    # Construct a unique experiment name
    EXPERIMENT_NAME="experiment_2_lr${lr}_bs${bs}"

    # Construct the command
    CMD="$BASE_CMD --name $EXPERIMENT_NAME --base configs/hyperparam_sweep/$config_filename --train --gpus 1"

    # Print the command (optional)
    echo "Running: $CMD"

    # Execute the command
    eval $CMD
done
