#!/bin/bash

# URL of the file to download

en_de="https://opus.nlpl.eu/download.php?f=ParaCrawl/v5/moses/de-en.txt.zip"
en_zh="http://opus.nlpl.eu/download.php?f=MultiUN/en-zh.txt.zip"
en_es="http://opus.nlpl.eu/download.php?f=MultiUN/en-es.txt.zip"
en_it="https://opus.nlpl.eu/download.php?f=Europarl/v3/moses/en-it.txt.zip"
en_fr="https://opus.nlpl.eu/download.php?f=ParaCrawl/v5/moses/en-fr.txt.zip"

urls=(${en_de} ${en_zh} ${en_es} ${en_it} ${en_fr})


# Get the absolute path of the directory containing the script
dir="$(cd "$(dirname "$0")" && pwd)"

# Destination parent directory for the downloaded file
data_dir="${dir}/data/Parallel_Corpus"

# Use curl to download the file and save it to the download_destination directory

for url in "${urls[@]}"; do
    # Extract the variable name from the URL
    filename=$(basename "${url}")
    zip_path="${data_dir}/${filename}"
    final_dest="${data_dir}/${filename%.*}"

    curl -L --output "${zip_path}" "${url}"
    unzip "${zip_path}" -d "${final_dest}"
    rm "${zip_path}"
done
