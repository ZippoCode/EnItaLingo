#!/bin/bash

ENV_NAME=".venv"
DATASET_DIR="datasets"
CHECKPOINTS_DIR="checkpoints"

# Create the virtual environment (overwrite if it exists)
python3 -m venv $ENV_NAME
echo "Virtual environment $ENV_NAME created."
source $ENV_NAME/bin/activate
echo "Virtual environment $ENV_NAME activated."

# Install the necessary packages
pip install -r requirements.txt
echo "Packages installed from requirements.txt."

# Install tensorflow-metal plug-in
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install tensorflow-metal==0.4.0
        echo "Plug-in tensorflow-metal installed."
fi

# Create the directories if it doesn't exist
mkdir -p $DATASET_DIR
mkdir -p $CHECKPOINTS_DIR
echo "Directories created or already exists."
