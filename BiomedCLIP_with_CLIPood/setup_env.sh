#!/bin/bash

# Define environment name
ENV_NAME="clipood_env"
PYTHON_VERSION="3.10"
REQUIREMENTS_FILE="requirements.txt"

# Define Miniconda installer URL
MINICONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"

# Step 1: Download Miniconda if not installed
if ! command -v conda &> /dev/null
then
    echo "🔍 Conda not found. Installing Miniconda..."
    wget $MINICONDA_URL -O $MINICONDA_INSTALLER
    bash $MINICONDA_INSTALLER -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
else
    echo "✅ Conda is already installed."
    eval "$(conda shell.bash hook)"
fi

# Step 2: Create environment
echo "🔧 Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Step 3: Activate environment
echo "⚙️  Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Step 4: Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "📦 Installing dependencies from $REQUIREMENTS_FILE"
    pip install --upgrade pip
    pip install -r $REQUIREMENTS_FILE
else
    echo "⚠️  Requirements file '$REQUIREMENTS_FILE' not found. Skipping dependency installation."
fi

echo "✅ Setup complete. Run 'conda activate $ENV_NAME' to activate your environment."
