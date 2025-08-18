#!/bin/bash

# Get the absolute path of the current script
current_dir=$(dirname "$(readlink -f "$0")")
scripts_dir="${current_dir}/scripts"

cd "${scripts_dir}" || exit
echo "Go to the directory: ${PWD}"

python3 preprocess_ucca.py --mrp_dir "${current_dir}/data/UCCA/mrp/2019/training/ucca" --xml_dir "${current_dir}/data/UCCA/UCCA_English-LPP-main/xml"
