#!/bin/bash

# Get the absolute path of the current script
current_dir=$(dirname "$(readlink -f "$0")")
scripts_dir="${current_dir}/scripts"

# add AMR directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${current_dir}/AMR"

cd "${scripts_dir}" || exit
echo "Go to the directory: ${PWD}"
python run_test.py

