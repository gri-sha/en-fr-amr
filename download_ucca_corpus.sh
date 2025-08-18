#!/bin/bash

ucca_train="http://svn.nlpl.eu/mrp/2019/public/ucca.tgz?p=28479"
ucca_test="https://codeload.github.com/UniversalConceptualCognitiveAnnotation/UCCA_English-LPP/zip/refs/heads/main"

dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="${dir}/data/UCCA"
mkdir -p "$data_dir"

train_dest="${data_dir}/mrp.tgz"
test_dest="${data_dir}/test.zip"

curl -L -o "${train_dest}" "${ucca_train}"
curl -L -o "${test_dest}" "${ucca_test}"

tar -xvf "${train_dest}" -C "${data_dir}"
unzip -o "${test_dest}" -d "${data_dir}"

rm "${train_dest}"
rm "${test_dest}"