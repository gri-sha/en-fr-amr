import numpy as np
import random
from numpy.random.mtrand import sample
from torch.utils.data import DataLoader
import torch
import logging
from torch.utils.data import DataLoader
from typing import Dict
# import internal package
from Collator import Collator
from settings import UCCA_DATA, AMR_DIR,TRN_MT_CORPUS, DEV_MT_CORPUS, mbart_lang_code_maps, semantic_graph_code_maps, mbart_special_token_dict
from text_utils import ucca_to_dict, amr_to_dict

class IterTrainDataset(torch.utils.data.IterableDataset) :

    def __init__(self, tokenizer, type):

        ucca_dir = UCCA_DATA
        amr_dir = AMR_DIR / "train" / "en"
        self.tokenizer = tokenizer
        self.type = type # training type "multilingual" or "bilingual"
        ucca = ucca_to_dict(ucca_dir, "train")
        amr = amr_to_dict(amr_dir, "train")
        graphs = ucca + amr
        self.concat_amr_ucca = list(map(self._preprocess_graphs, graphs))

        if self.type == "multilingual":
            self.en_es = {"en": open(TRN_MT_CORPUS["en-es.en"]).readlines(),
                          "es": open(TRN_MT_CORPUS["en-es.es"]).readlines()}
            self.en_de = {"en": open(TRN_MT_CORPUS["en-de.en"]).readlines(),
                          "de": open(TRN_MT_CORPUS["en-de.de"]).readlines()}
            self.en_fr = {"en": open(TRN_MT_CORPUS["en-fr.en"]).readlines(),
                          "fr": open(TRN_MT_CORPUS["en-fr.fr"]).readlines()}
            self.en_it = {"en": open(TRN_MT_CORPUS["en-it.en"]).readlines(),
                          "it": open(TRN_MT_CORPUS["en-it.it"]).readlines()}
            self.en_zh = {"en": open(TRN_MT_CORPUS["en-zh.en"]).readlines(),
                          "zh": open(TRN_MT_CORPUS["en-zh.zh"]).readlines()}

            self.datasets = [self.concat_amr_ucca, self.en_es, self.en_de, self.en_it, self.en_zh, self.en_fr]
            self.ds_probability = [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]

        elif self.type =="bilingual":
            self.en_fr = {"en": open(TRN_MT_CORPUS["en-fr.en"]).readlines(),
                          "fr": open(TRN_MT_CORPUS["en-fr.fr"]).readlines()}
            self.datasets = [self.concat_amr_ucca, self.en_fr]
            self.ds_probability = [0.8, 0.2]


        self.amr_ucca_ds_idx = 0


    def __iter__(self):
        while True: # keeps yielding new datapoint while training
            ds_index = np.random.choice(range(0, len(self.datasets)), size=1, p=self.ds_probability)[0]
            if ds_index == self.amr_ucca_ds_idx:
                random_idx = random.sample(range(len(self.concat_amr_ucca)), 1)[0]
                yield self.concat_amr_ucca[random_idx]
            else:
                random_idx = random.sample(range(len(self.datasets[ds_index]["en"])), 1)[0] # get random index using the total length of english side corpus
                sampled_translation = {lang: self.datasets[ds_index][lang][random_idx] for lang in self.datasets[ds_index]}
                encoded_translation = self._preprocess_texts(sampled_translation)
                yield encoded_translation

    def _preprocess_texts(self, data:Dict[str, str]):

        # define source and target language for the tokenization
        lang_set = list(data.keys())
        random.shuffle(lang_set)  # randomly shuffle lang set to flip translation directions
        source_lang, target_lang = lang_set
        tokenizer = self.tokenizer

        # set source language and target language for tokenizer
        tokenizer.src_lang = mbart_lang_code_maps[source_lang]
        tokenizer.tgt_lang = mbart_lang_code_maps[target_lang]

        # tokenizer input and labels
        model_inputs = tokenizer(data[source_lang], truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(data[target_lang], truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def _preprocess_graphs(self, data:Dict[str, str]):

        source_lang = "en"
        target_graph_name = ''.join(set(data.keys()) & set(semantic_graph_code_maps.keys()))  # returns a string 'amr' or 'ucca'
        tokenizer = self.tokenizer

        tokenizer.src_lang = mbart_lang_code_maps[source_lang]
        tokenizer.tgt_lang = semantic_graph_code_maps[target_graph_name]

        model_inputs = tokenizer(data[source_lang], truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(data[target_graph_name], truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

class EvalDataset(torch.utils.data.Dataset):
    """
    Eval dataset for semantic parsing : ["ucca", "amr"]
    """
    def __init__(self, dataset, tokenizer, graph_name):
        self.graph_name = graph_name
        self.tokenizer = tokenizer
        self.dataset = list(map(self.preprocess_dev_graphs, dataset))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def preprocess_dev_graphs(self, data:Dict[str, str]):
        source_lang = ''.join(set(data.keys()) & set(mbart_lang_code_maps.keys()))
        target_graph_name = self.graph_name
        tokenizer = self.tokenizer
        tokenizer.src_lang = mbart_lang_code_maps[source_lang]
        tokenizer.tgt_lang = semantic_graph_code_maps[target_graph_name]
        model_inputs = tokenizer(data[source_lang], truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(data[target_graph_name], truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


class EvalMTDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, src_lang):

        self.src_lang = src_lang
        self.tgt_lang = "en"

        src_txt_file = DEV_MT_CORPUS["{}-{}.{}".format(self.tgt_lang, self.src_lang, self.src_lang)]
        tgt_txt_file = DEV_MT_CORPUS["{}-{}.{}".format(self.tgt_lang, self.src_lang, self.tgt_lang)]

        src_texts = open(src_txt_file).readlines()
        tgt_texts = open(tgt_txt_file).readlines()

        parallel_texts = [{self.src_lang: src_text,
                           self.tgt_lang: tgt_text}
                          for src_text, tgt_text in zip(src_texts, tgt_texts) ]

        self.tokenizer = tokenizer
        self.dataset = list(map(self._preprocess_texts, parallel_texts))


    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _preprocess_texts(self, data: Dict[str, str]):

        tokenizer = self.tokenizer
        source_lang = self.src_lang
        target_lang = self.tgt_lang

        tokenizer.src_lang = mbart_lang_code_maps[source_lang]
        tokenizer.tgt_lang = mbart_lang_code_maps[target_lang]

        model_inputs = tokenizer(data[source_lang], truncation=True, return_tensors='pt')

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(data[target_lang], truncation=True, return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

if __name__ == '__main__':

    from transformers import AutoTokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    ucca_dir = UCCA_DATA
    amr_dir = AMR_DIR
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="en_XX")
    tokenizer.add_special_tokens(mbart_special_token_dict)
    train_set = IterTrainDataset(tokenizer, "multilingual")
    collate_fn = Collator(device)
    trn_dataloader = DataLoader(train_set, batch_size=4, collate_fn=collate_fn, prefetch_factor=2)
    total_batch = 0
    total_n_step = (128 // 4) * 10000

    for i in range(total_n_step):
        trn_input = next(iter(trn_dataloader))
        input_txt = tokenizer.batch_decode(trn_input['input_ids'], skip_special_tokens=True)
        labels = []
        for i in range(len(trn_input['input_ids'])):
            label_id = i
            try:
                neg_index = (trn_input['labels'][label_id] == -100).nonzero(as_tuple=True)[0][0].item()
                label = tokenizer.decode(trn_input["labels"][label_id][:neg_index])
                labels.append(label)
            except:
                label = tokenizer.decode(trn_input["labels"][label_id])
                labels.append(label)
                continue
        for i, (input, label) in enumerate(zip(input_txt, labels)):
            print("input [{}]: {}".format(i + 1, input))
            print("label [{}]: {}".format(i + 1, label))
        print("\n======================================\n")

    # check since which step, it gets empty input
    # accum_steps = 128 // 4
    # total_n_step = 50000 * accum_steps
    # for j in range(total_n_step):
    #     trn_input = next(iter(trn_dataloader))
    #     if j % (1000 * accum_steps) == 0:
    #         print("step {}".format(j / accum_steps))
    #     if any(trn_input['input_ids'][:,0]==2):
    #         print("===== empty input at step {}".format(j))
    #         input_txt = tokenizer.batch_decode(trn_input['input_ids'], skip_special_tokens=True)
    #         labels = []
    #         for i in range(len(trn_input['input_ids'])):
    #             label_id = i
    #             try:
    #                 neg_index = (trn_input['labels'][label_id] == -100).nonzero(as_tuple=True)[0][0].item()
    #                 label = tokenizer.decode(trn_input["labels"][label_id][:neg_index])
    #                 labels.append(label)
    #             except:
    #                 label = tokenizer.decode(trn_input["labels"][label_id])
    #                 labels.append(label)
    #                 continue
    #         for i, (input, label) in enumerate(zip(input_txt, labels)):
    #             print("input [{}]: {}".format(i + 1, input))
    #             print("label [{}]: {}".format(i + 1, label))
    #         print("\n======================================\n")

    collator = Collator(device=device, pad_labels=True)
    amr_dev_ml_dataloaders = dict()
    amr_src_langs = [lang for lang in ["es"]]
    for src_lang in amr_src_langs:
        amr_dir = AMR_DIR / "dev" / "{}".format(src_lang)
        amr_dev_ml = amr_to_dict(amr_dir, "800.dev")
        amr_dev_ds_ml = EvalDataset(amr_dev_ml, tokenizer, 'amr')
        dev_graph_dataloader_ml = DataLoader(amr_dev_ds_ml, batch_size=8, collate_fn=collator)
        amr_dev_ml_dataloaders[src_lang] = dev_graph_dataloader_ml

    dl = amr_dev_ml_dataloaders["es"]
    for j in dl:
        labels = []
        trn_input = j
        input_txt = tokenizer.batch_decode(trn_input['input_ids'], skip_special_tokens=True)
        for i in range(len(trn_input['input_ids'])):
            label_id = i
            try:
                neg_index = (trn_input['labels'][label_id] == -100).nonzero(as_tuple=True)[0][0].item()
                label = tokenizer.decode(trn_input["labels"][label_id][:neg_index])
                labels.append(label)
            except:
                label = tokenizer.decode(trn_input["labels"][label_id])
                labels.append(label)
                continue
        for i, (input, label) in enumerate(zip(input_txt, labels)):
            print("input [{}]: {}".format(i + 1, input))
            print("label [{}]: {}".format(i + 1, label))
        print("\n======================================\n")