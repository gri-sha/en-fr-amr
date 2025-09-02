#!/bin/bash

# Get the absolute path of the current script
current_dir=$(dirname "$(readlink -f "$0")")
scripts_dir="${current_dir}/scripts"

cd "${scripts_dir}" || exit
echo "Go to the directory: ${PWD}"

python3 preprocess_parallel_corpus.py -en ../data/Parallel_Corpus/de-en.txt/ParaCrawl.de-en.en --tgt ../data/Parallel_Corpus/de-en.txt/ParaCrawl.de-en.de
python3 preprocess_parallel_corpus.py -en ../data/Parallel_Corpus/en-zh.txt/MultiUN.en-zh.en --tgt ../data/Parallel_Corpus/en-zh.txt/MultiUN.en-zh.zh --char_ratio_threshold 5.0
python3 preprocess_parallel_corpus.py -en ../data/Parallel_Corpus/en-es.txt/MultiUN.en-es.en --tgt ../data/Parallel_Corpus/en-es.txt/MultiUN.en-es.es
python3 preprocess_parallel_corpus.py -en ../data/Parallel_Corpus/en-it.txt/Europarl.en-it.en --tgt ../data/Parallel_Corpus/en-it.txt/Europarl.en-it.it
python3 preprocess_parallel_corpus.py -en ../data/Parallel_Corpus/en-fr.txt/ParaCrawl.en-fr.en --tgt ../data/Parallel_Corpus/en-fr.txt/ParaCrawl.en-fr.fr