import torch
import logging
from torch.utils.data import DataLoader
from CustomDatasets import EvalDataset
from transformers import MBartForConditionalGeneration, AutoConfig, AutoTokenizer
from pathlib import Path
# import internal package
from text_utils import amr_to_dict, ucca_to_dict
import argparse
from settings import AMR_DIR, TEMP, MODEL, ROOT, DATA, mbart_lang_code_maps, mbart_special_token_dict, UCCA_DATA
from Evaluator import AmrEvaluator, UccaEvaluator
from Collator import Collator

def run_test(bestmodel, task_name, gold_graph_file, ref_sent_file, log, batch_size):
    task_name = task_name.lower()
    log.info("evaluating on {}".format(task_name))
    checkpoint = torch.load(bestmodel)
    prediction_save_to = TEMP / args.best_model / "temp_test_{}".format(task_name)

    amr_dir = AMR_DIR / "test" / "{}".format(task_name)
    log.info(gold_graph_file)
    log.info(ref_sent_file)

    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    tokenizer.add_special_tokens(mbart_special_token_dict)

    src_lang = mbart_lang_code_maps[task_name] if task_name != "ucca" else "en_XX"
    tgt_lang = "amr" if task_name != "ucca" else "ucca"

    model.resize_token_embeddings(len(tokenizer))

    if task_name == "ucca":
        data_dict = ucca_to_dict(UCCA_DATA, "test")
        dataset = EvalDataset(data_dict, tokenizer, "ucca")
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        evaluator = UccaEvaluator(tokenizer, gold_graph_file, ref_sent_file, TEMP, model, dataloader, log, split="test")

    else:
        data_dict = amr_to_dict(amr_dir, split="test")
        dataset = EvalDataset(data_dict, tokenizer, 'amr')
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        evaluator = AmrEvaluator(tokenizer, gold_graph_file, ref_sent_file, prediction_save_to, model, dataloader,
                                 log, src_lang=src_lang[:2])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    loss, accuracy = evaluator.run_eval("test")

    log.info(f"{src_lang}-{tgt_lang} loss: {loss}, accuracy: {accuracy}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-cc25")
    parser.add_argument('--best_model', type=str, default="mbart_mt_best-model-checkpoint.pt", help="best model file e.g. best_model.pt (the checkpoint shoud be in models directory)")
    parser.add_argument('--task_names', type=str, default=["en", "de", "fr", "it", "es", "ucca"], nargs='+')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    BEST_MODEL = MODEL / args.best_model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    config = AutoConfig.from_pretrained("facebook/mbart-large-cc25")
    model = MBartForConditionalGeneration(config)

    checkpoint = torch.load(BEST_MODEL)
    collate_fn = Collator(device)

    for task_name in args.task_names:
        if task_name == "ucca":
            gold_graph_file = UCCA_DATA / "test.tf"
            ref_sent_file = UCCA_DATA / "test.sent"
        else:
            gold_graph_file = AMR_DIR / "test" / "{}".format(task_name) /  "test.txt.graph"
            ref_sent_file = AMR_DIR / "test" / "{}".format(task_name) /   "test.txt.sent"

        run_test(BEST_MODEL, task_name, gold_graph_file, ref_sent_file, log, batch_size=args.batch_size)

"""
        for l in langs:
            l = l.lower()
            log.info("evaluating on {}".format(l))
            TEMP = ROOT / "TEMP" / args.best_model / "temp_test_{}".format(l)
            prediction_save_to = TEMP
            amr_dir = AMR_DIR / "test" / "{}".format(l)
            eval_gold_file = amr_dir / "test.txt.graph"
            eval_sent_file = amr_dir / "test.txt.sent"
            log.info(eval_gold_file)
            log.info(eval_sent_file)
            src_lang = mbart_lang_code_maps[l]
            tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang=src_lang, tgt_lang="en_XX")
            tokenizer.add_special_tokens(mbart_special_token_dict)
            model.resize_token_embeddings(len(tokenizer))
            amr_test = amr_to_dict(amr_dir, split="test")
            amr_dev_ds = EvalDataset(amr_test, tokenizer, 'amr')
            amr_dataloader = DataLoader(amr_dev_ds, batch_size=batch_size, collate_fn=collate_fn, prefetch_factor=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            evaluator = AmrEvaluator(tokenizer, eval_gold_file, eval_sent_file, prediction_save_to, model, amr_dataloader, log, src_lang=src_lang[:2])
            loss, smatch = evaluator.run_eval("eval")

            log.info("smatch score for {}: {}".format(l, smatch))

    elif args.eval_lang=='en':
        TEMP = ROOT / "TEMP" / "temp_test_en"
        prediction_save_to = TEMP / args.best_model
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="en_XX")
        tokenizer.add_special_tokens(mbart_special_token_dict)
        model.resize_token_embeddings(len(tokenizer))
        amr_dir = AMR_DIR / "test" / "{}".format("en")
        eval_gold_file = amr_dir / "test.txt.graph"
        eval_sent_file = amr_dir / "test.txt.sent"
        amr_test = amr_to_dict(amr_dir, split="test")
        amr_dev_ds = EvalDataset(amr_test, tokenizer, 'amr')
        amr_dataloader = DataLoader(amr_dev_ds, batch_size=batch_size, collate_fn=collate_fn, prefetch_factor=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        evaluator = AmrEvaluator(tokenizer, eval_gold_file, eval_sent_file, prediction_save_to, model, amr_dataloader, log)
        loss, smatch = evaluator.run_eval("eval")
        log.info("smatch score for english: {}".format(smatch))

    elif args.eval_lang=='ucca':

        TEMP = ROOT / "TEMP" / args.best_model

        ucca = ucca_to_dict(UCCA_DATA, "test")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="en_XX", tgt_lang="en_XX")
        tokenizer.add_special_tokens(mbart_special_token_dict)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        generated = []
        ucca_ds = EvalDataset(ucca, tokenizer, "ucca")
        dev_ucca_dataloader = DataLoader(ucca_ds, batch_size=batch_size, collate_fn=collate_fn)
        decoder_start_token_id = tokenizer.convert_tokens_to_ids(["ucca"])[0]
        ucca_gold_file = UCCA_DATA / "test.tf"
        ucca_sent_file = UCCA_DATA / "test.sent"

        ucca_evaluator = UccaEvaluator(tokenizer, ucca_gold_file, ucca_sent_file, TEMP, model, dev_ucca_dataloader, log, is_test=True)
        eval_loss, ucca_score = ucca_evaluator.run_eval("eval")
        log.info("loss: {} ucca score: {}".format(eval_loss, ucca_score))
"""