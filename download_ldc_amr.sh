#!/bin/bash
set -e

source .env

filename="ldc_amr.tgz"
dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="${dir}/data/AMR"

# Download without virus scan
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}&confirm=t&no_cookies=1" -O "$filename"

mkdir -p "$data_dir"
tar -xzf "$filename" -C "$data_dir"

rm "$filename"

echo "Download and extraction complete."