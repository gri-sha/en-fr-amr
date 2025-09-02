#!/bin/bash

# Get the absolute path of the current script
current_dir=$(dirname "$(readlink -f "$0")")
scripts_dir="${current_dir}/scripts"

# add AMR directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${current_dir}/AMR"

source .venv/bin/activate

cd "${scripts_dir}" || exit
echo "Go to the directory: ${PWD}"
python train_amr_parser.py 
