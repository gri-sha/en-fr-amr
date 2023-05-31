#!/bin/bash

# URL of the file to download

ucca_train="http://svn.nlpl.eu/mrp/2019/public/ucca.tgz?p=28479"
ucca_test="https://codeload.github.com/UniversalConceptualCognitiveAnnotation/UCCA_English-LPP/zip/refs/heads/main"


# Get the absolute path of the directory containing the script
dir_name="$(cd "$(dirname "$0")" && pwd)"
echo $dir_name

# Destination directory for the downloaded file
train_dest="${dir_name}/data/UCCA/mrp.gz"
test_dest="${dir_name}/data/UCCA/test.zip"

echo $train_dest
echo $test_dest

curl --output "${train_dest}" -L "${ucca_train}"
curl --output "${test_dest}" -L "${ucca_test}"

tar -xvf "${train_dest}" -C "${dir_name}/data/UCCA"
unzip -o "${test_dest}" -d "${dir_name}/data/UCCA"

#rm "${test_dest}"
#rm "${train_dest}"
