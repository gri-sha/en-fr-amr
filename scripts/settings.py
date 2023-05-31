from pathlib import Path

CURR = Path().cwd()   # SGL/scripts
ROOT = CURR.parent    # SGL
TEMP = ROOT / "TEMP"
DATA = ROOT / "data"
PARALLEL_CORPUS_DIR = DATA / "Parallel_Corpus"
AMR_DIR = DATA / "AMR"
UCCA_DATA = DATA / "UCCA"
FR_SYNTAX = DATA / "Syntax" / "three_syntax_corpus_single_spaced"
MODEL = ROOT / "models"
AMR_SCRIPT = ROOT / "AMR"
CHINESE_VOCAB = CURR / "chinese_vocab.txt"
CHINESE_VOCAB_PAD = CURR / "chinese_vocab_padded_front.txt"

TRN_MT_CORPUS = {"en-es.en": PARALLEL_CORPUS_DIR / "train" / "en-es.txt" / "en-es.en",
                 "en-es.es": PARALLEL_CORPUS_DIR / "train" / "en-es.txt" / "en-es.es",
                 "en-de.en": PARALLEL_CORPUS_DIR / "train" / "en-de.txt" / "en-de.en",
                 "en-de.de": PARALLEL_CORPUS_DIR / "train" / "en-de.txt" / "en-de.de",
                 "en-fr.en": PARALLEL_CORPUS_DIR / "train" / "en-fr.txt" / "en-fr.en",
                 "en-fr.fr": PARALLEL_CORPUS_DIR / "train" / "en-fr.txt" / "en-fr.fr",
                 "en-it.en": PARALLEL_CORPUS_DIR / "train" / "en-it.txt" / "en-it.en",
                 "en-it.it": PARALLEL_CORPUS_DIR / "train" / "en-it.txt" / "en-it.it",
                 "en-zh.en": PARALLEL_CORPUS_DIR / "train" / "en-zh.txt" / "en-zh.en",
                 "en-zh.zh": PARALLEL_CORPUS_DIR / "train" / "en-zh.txt" / "en-zh.zh"}


DEV_MT_CORPUS = {"en-es.en": PARALLEL_CORPUS_DIR / "dev" / "en-es.txt" / "en-es.en",
                 "en-es.es": PARALLEL_CORPUS_DIR / "dev" / "en-es.txt" / "en-es.es",
                 "en-de.en": PARALLEL_CORPUS_DIR / "dev" / "en-de.txt" / "en-de.en",
                 "en-de.de": PARALLEL_CORPUS_DIR / "dev" / "en-de.txt" / "en-de.de",
                 "en-fr.en": PARALLEL_CORPUS_DIR / "dev" / "en-fr.txt" / "en-fr.en",
                 "en-fr.fr": PARALLEL_CORPUS_DIR / "dev" / "en-fr.txt" / "en-fr.fr",
                 "en-it.en": PARALLEL_CORPUS_DIR / "dev" / "en-it.txt" / "en-it.en",
                 "en-it.it": PARALLEL_CORPUS_DIR / "dev" / "en-it.txt" / "en-it.it",
                 "en-zh.en": PARALLEL_CORPUS_DIR / "dev" / "en-zh.txt" / "en-zh.en",
                 "en-zh.zh": PARALLEL_CORPUS_DIR / "dev" / "en-zh.txt" / "en-zh.zh"}

mbart_lang_code_maps = {'en': 'en_XX',
                        'it': 'it_IT',
                        'de': 'de_DE',
                        'es': 'es_XX',
                        'zh': 'zh_CN',
                        'fr': 'fr_XX'}

semantic_graph_code_maps = {'amr': 'amr',
                            'ucca': 'ucca'}

mbart_special_token_dict = {'additional_special_tokens': ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX',
                                                            'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN',
                                                            'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT',
                                                            'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO',
                                                            'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN',
                                                             'amr', 'ucca']
                            }