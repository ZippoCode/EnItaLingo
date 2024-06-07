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
    pip install tensorflow-metal
        echo "Plug-in tensorflow-metal installed."
fi

# Download Models & Languages
if python -c "import spacy; spacy.load('en_core_web_sm')" &> /dev/null; then
    echo "spaCy model 'en_core_web_sm' is already installed."
else
    # Install spaCy model
    python -m spacy download en_core_web_sm
    echo "spaCy model 'en_core_web_sm' downloaded."
fi
if python -c "import spacy; spacy.load('it_core_news_sm')" &> /dev/null; then
    echo "spaCy model 'it_core_news_sm' is already installed."
else
    # Install spaCy model
    python -m spacy download it_core_news_sm
    echo "spaCy model 'it_core_news_sm' downloaded."
fi
echo "Languages installed."

# Create the directories if it doesn't exist
mkdir -p $DATASET_DIR
mkdir -p $CHECKPOINTS_DIR
echo "Directories created or already exists."
