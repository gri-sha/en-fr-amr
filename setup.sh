#!/bin/bash
set -e

# Set up Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Get an preprocess data
bash install_dependencies.sh
bash download_ldc_amr.sh
bash preprocess_ldc_amr.sh
bash download_ucca_corpus.sh
bash preprocess_ucca.sh
bash download_parallel_corpus.sh
bash preprocess_parallel_corpus.sh