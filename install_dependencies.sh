#!/bin/bash

# Initial setup of the project environment.

# Set up Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Clone AMR repository
echo "Cloning AMR repository..."
git clone https://github.com/RikVN/AMR.git

# Create data directory (if it doesn't exist)
mkdir -p data
