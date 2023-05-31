from pathlib import Path
import time
from typing import List
from settings import PARALLEL_CORPUS_DIR
from datetime import timedelta

def file_to_list(filepath1, filepath2):

    with open(filepath1) as f1, open(filepath2) as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        zipped = list(zip(file1_lines, file2_lines))
        return zipped

def check_minimum_length(bisentence):

    snt1, snt2 = bisentence

    if len(snt1) < 25:
        return False
    if len(snt2) < 25:
        return False
    if len(snt2) > 1000 :
        return False
    if len(snt2) > 1000:
        return False

    return True

def check_char_ratio(bisentence):

    snt1, snt2 = bisentence
    short_len, long_len = sorted([len(snt1), len(snt2)])
    if long_len / short_len < char_ratio_threshold:
        return True
    else:
        return False


def del_idx(prl_snt, idx):
    indexes = idx

    for index in sorted(indexes, reverse=True):
        del prl_snt[index]
    return prl_snt


def save_to_file(prl_snt: List, src_lang, tgt_lang, split_name):

    save_to_dir = PARALLEL_CORPUS_DIR / split_name / "{}-{}.txt".format(src_lang, tgt_lang)
    save_to_dir.mkdir(parents=True, exist_ok=True)

    src_file = save_to_dir / "{}-{}.{}".format(src_lang, tgt_lang, src_lang)
    tgt_file = save_to_dir / "{}-{}.{}".format(src_lang, tgt_lang, tgt_lang)

    lang1_snt, lang2_snt = list(zip(*prl_snt))
    with open(src_file, 'w') as f1, open(tgt_file, 'w') as f2:
        f1.writelines(lang1_snt)
        f2.writelines(lang2_snt)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--english_file', required=True, help="Name of the english text file ex) dir_name /  MultiUN.en-zh.en")
    parser.add_argument('-tgt', '--tgt_file', required=True, help="Name of the target text file ex) dir_name / MultiUN.en-zh.zh")
    parser.add_argument('--char_ratio_threshold', type=float, default=3.0, help="Threshold for char ratio")

    args = parser.parse_args()
    # print the whole args dict
    print(args.__dict__)
    src_file = Path(args.english_file)
    tgt_file = Path(args.tgt_file)
    char_ratio_threshold = args.char_ratio_threshold
    src_lang = src_file.suffix.split('.')[-1]
    tgt_lang = tgt_file.suffix.split('.')[-1]
    parallel_sentences = file_to_list(src_file, tgt_file)

    # 1. filter by min or max length of sentence
    filtered_by_length = list(filter(check_minimum_length, parallel_sentences))
    print('{} / {} examples filtered by length'.format(len(parallel_sentences) - len(filtered_by_length), len(parallel_sentences)))

    # 2. filter by char ratio between src and tgt
    filtered_by_char_ratio = list(filter(check_char_ratio, filtered_by_length))
    print('{} / {} examples filtered by char ratio'.format(len(filtered_by_length) - len(filtered_by_char_ratio), len(parallel_sentences)))

    # 3. get dev set from filtered corpus
    train_n_lines = 5000000
    dev_n_lines = 400

    filtered_by_order_train = filtered_by_char_ratio[:train_n_lines]
    filtered_by_order_dev = filtered_by_char_ratio[-dev_n_lines:]

    save_to_file(filtered_by_order_train, src_lang, tgt_lang, split_name="train")
    save_to_file(filtered_by_order_dev, src_lang, tgt_lang, split_name="dev")
