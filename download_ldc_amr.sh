#!/bin/bash
set -e

source .env

FILE_NAME="ldc_amr.tgz"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/"

# Download without virus scan
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}&confirm=t&no_cookies=1" -O "$FILE_NAME"

mkdir -p "$DATA_DIR"
tar -xzf "$FILE_NAME" -C "$DATA_DIR"

rm "$FILE_NAME"

echo "Download and extraction complete."