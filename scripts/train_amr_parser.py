import wandb
import torch
import numpy as np
from transformers import AutoConfig
from transformers import get_scheduler
import logging
from itertools import islice
from tqdm.auto import tqdm
from torch.optim import AdamW
import argparse
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, AutoTokenizer
# import internal package
from Collator import Collator
from settings import TEMP, ROOT, MODEL, UCCA_DATA, AMR_DIR, mbart_lang_code_maps, mbart_special_token_dict
from pytorchtools import EarlyStopping
from Evaluator import AmrEvaluator, MTEvaluator, UccaEvaluator
from CustomDatasets import IterTrainDataset, EvalDataset, EvalMTDataset
from text_utils import ucca_to_dict, amr_to_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(trn_dataloader,
          model,
          optimizer,
          dev_dataloader,
          dev_ucca_dataloader,
          max_steps,
          tokenizer,
          lr_scheduler,
          eval_freq,
          early_stop_patience,
          grad_accum_steps):


    start_step = 0

    if wandb.run.resumed and "_step" in wandb.summary.keys():
        start_step = wandb.summary["_step"]

    ucca_gold_file = UCCA_DATA / "dev.tf"
    ucca_sent_file = UCCA_DATA / "dev.sent"
    ucca_pred_save_dir = SAVE_PRED_TO

    model.to(device)
    num_training_steps = max_steps
    early_stopping = EarlyStopping(history=EARLY_STOP,
                                   patience=(early_stop_patience // eval_freq),
                                   verbose=True, best_model_save_to=BEST_MODEL,
                                   last_checkpoint_save_to=LAST_CHECKPOINT,
                                   trace_func=log.info)

    dev_en_amr_graph = AMR_DIR / "dev" / "en" / "dev.txt.graph"
    dev_en_amr_sent = AMR_DIR / "dev" / "en" / "dev.txt.sent"
    pred_save_dir = SAVE_PRED_TO


    for current_step in tqdm(range(start_step, num_training_steps), initial=start_step, total=num_training_steps):
        """
        Training 
        """
        model.train()
        model.zero_grad()
        trn_losses = []
        counter = 0

        for trn_input in islice(trn_dataloader, grad_accum_steps):
            outputs = model(**trn_input)
            loss = outputs.loss / grad_accum_steps
            trn_losses.append(outputs.loss.item())
            loss.backward()  # gradient compute & stored
            counter += 1

        trn_loss = np.mean(trn_losses)
        log.info("current learning rate: {}".format(optimizer.param_groups[0]['lr']))
        optimizer.step()    # weight update
        optimizer.zero_grad(set_to_none=True)

        lr_scheduler.step()

        log.info(" train loss: {}".format(trn_loss))
        wandb.log({"train loss": trn_loss}, step=current_step)

        if current_step % eval_freq == 0 and current_step != 0:

            # eval on ucca dev
            ucca_evaluator = UccaEvaluator(tokenizer, ucca_gold_file, ucca_sent_file,
                                           ucca_pred_save_dir, model, dev_ucca_dataloader, log, split="dev")
            eval_ucca_loss, eval_ucca_score = ucca_evaluator.run_eval()

            wandb.log({"dev_ucca_score": eval_ucca_score, "dev_ucca_loss": eval_ucca_loss}, step=current_step)
            log.info("\nstep:{} - ucca dev loss: {}, ucca dev score: {}".format(current_step, eval_ucca_loss, eval_ucca_score))

            # eval on amr dev (early stopping based on amr dev)
            amr_evaluator = AmrEvaluator(tokenizer, dev_en_amr_graph, dev_en_amr_sent,
                                         pred_save_dir, model, dev_dataloader, log)
            eval_amr_loss, eval_smatch = amr_evaluator.run_eval(current_step)

            wandb.log({"dev_smatch": eval_smatch, "dev loss": eval_amr_loss}, step=current_step)
            log.info("\nstep:{} - dev loss: {}, dev smatch: {}".format(current_step, eval_amr_loss, eval_smatch))

            early_stopping(model=model, val_accuracy=eval_smatch, optimizer=optimizer, lr_scheduler=lr_scheduler)

            if early_stopping.early_stop:
                wandb.finish()
                return

    wandb.finish()
    return

def load_models(tokenizer, num_warmup_steps, model_name, dropout, learning_rate, num_training_steps):

    if BEST_MODEL.exists() or LAST_CHECKPOINT.exists(): #TODO: no need to load BEST_MODEL
        try:
            checkpoint = torch.load(LAST_CHECKPOINT)
            log.info("Succesfully loaded the last checkpoint from {}".format(LAST_CHECKPOINT))
        except:
            checkpoint = torch.load(BEST_MODEL)
            log.info("Last checkpoint not found, best checkpoint loaded from {}".format(BEST_MODEL))

        config = AutoConfig.from_pretrained(model_name, dropout=dropout)
        model = MBartForConditionalGeneration(config)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    else:
        log.info("no checkpoint.. initializing model")
        model = MBartForConditionalGeneration.from_pretrained(model_name, dropout=dropout)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    return model, optimizer, lr_scheduler

def load_data(tokenizer, multi_or_bi, train_batch_size, eval_batch_size):
    ucca_dir = UCCA_DATA
    amr_dir = AMR_DIR / "dev" / "en"

    collator = Collator(device=device, pad_labels=True)

    train_set = IterTrainDataset(tokenizer, type=multi_or_bi) # contains amr + ucca + mt data
    trn_dataloader = DataLoader(train_set, batch_size=train_batch_size, collate_fn=collator)

    ucca_dev = ucca_to_dict(ucca_dir, "dev")
    amr_dev = amr_to_dict(amr_dir, "dev")
    ucca_dev_ds = EvalDataset(ucca_dev, tokenizer, 'ucca') #TODO
    amr_dev_ds = EvalDataset(amr_dev, tokenizer, 'amr')
    dev_ucca_dataloader = DataLoader(ucca_dev_ds, batch_size=eval_batch_size, collate_fn=collator)
    dev_amr_dataloader = DataLoader(amr_dev_ds, batch_size=eval_batch_size, collate_fn=collator)

    return trn_dataloader, dev_amr_dataloader, dev_ucca_dataloader


def main(**kwargs):
    print(kwargs)
    tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name'])
    tokenizer.add_special_tokens(mbart_special_token_dict)

    trn_dataloader, dev_dataloader, dev_ucca_dataloader = load_data(tokenizer,
                                                                    kwargs["multi_or_bi"],
                                                                    kwargs["train_batch_size"],
                                                                    kwargs["eval_batch_size"])
    model, optimizer, lr_scheduler = load_models(tokenizer,
                                                 num_warmup_steps=kwargs["num_warmup_steps"],
                                                 model_name=kwargs["model_name"],
                                                 dropout=kwargs["dropout"],
                                                 learning_rate=kwargs["learning_rate"],
                                                 num_training_steps=kwargs["num_training_steps"])
    log.info(model.config)
    log.info("training starts...")
    train(trn_dataloader=trn_dataloader,
          model=model,
          optimizer=optimizer,
          dev_dataloader=dev_dataloader,
          dev_ucca_dataloader=dev_ucca_dataloader,
          max_steps=kwargs["num_training_steps"],
          tokenizer=tokenizer,
          lr_scheduler=lr_scheduler,
          eval_freq=kwargs["eval_freq"],
          early_stop_patience=kwargs["early_stop_patience"],
          grad_accum_steps=kwargs["grad_accum_steps"],
          )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--resume_id', type=str, help='get wandb id to resume training (optional)')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--grad_accum_steps', type=int, default=30, help='gradient_accumulation_steps')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--num_training_steps', type=int, default=300000, help='a total number of training steps')
    parser.add_argument('--eval_freq', type=int, default=1000, help='evaluate every n steps')
    parser.add_argument('--num_warmup_steps', type=int, default=2500, help='warmup steps')
    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-cc25", help='model name')
    parser.add_argument('--early_stop_patience', type=int, default=25000, help='early stop patience')
    parser.add_argument('--multi_or_bi', type=str, choices=["multilingual", "bilingual"], default="multilingual", help='multilingual or bilingual')

    args = parser.parse_args()

    if args.resume_id is None:
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, resume="allow", config=args.__dict__)
    else:
        wandb.init(id=args.resume_id, resume="allow", config=args.__dict__)

    BEST_MODEL = MODEL / "{}-best-model-checkpoint.pt".format(wandb.run.id)
    LAST_CHECKPOINT = MODEL / "{}-last-checkpoint.pt".format(wandb.run.id)
    SAVE_PRED_TO = TEMP / "output_{}".format(wandb.run.id)
    EARLY_STOP = ROOT / "early_stop_variables" / "{}_earlystop_var.pk".format(wandb.run.id)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    log.info(args.__dict__)

    main(**vars(args))
