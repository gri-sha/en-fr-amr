import numpy as np
import torch
import pickle
from pathlib import Path

"""
Original source code: 
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py 
"""

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, history,
                 patience=20,
                 verbose=False,
                 delta=0,
                 best_model_save_to='best-model-checkpoint.pt',
                 last_checkpoint_save_to='last-checkpoint.pt',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            best_model_save_to (str): Path for the best checkpoint to be saved to.
                            Default: 'best-model-checkpoint.pt'
            last_checkpoint_save_to (str): Path for the last checkpoint to be saved to.
                            Default: 'last-checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.pickled = Path(history)
        self.early_stop = False
        self.val_accuracy_max = np.Inf   # bests_core before updating best_score
        self.delta = delta
        self.best_model_save_to = best_model_save_to
        self.last_checkpoint_save_to = last_checkpoint_save_to
        self.trace_func = trace_func
        if self.pickled.exists():
            with open(self.pickled, 'rb') as pck:
                pickled = pickle.load(pck)
                self.counter = pickled["es_counter"]
                self.best_score = pickled["es_best_score"]
                trace_func("loading early stopping info from last checkpoint: counter: {}, best_score:{}".format(self.counter, self.best_score))
        else:
            self.counter = 0
            self.best_score = -0.01 # to forthe the saving of the model when the smatch is 0 at the beginning
            trace_func("Initiate early stopping: counter: {}, best_score:{}".format(self.counter, self.best_score))

    def __call__(self, val_accuracy, model, optimizer, lr_scheduler):

        # score = -val_loss  # why minus ?
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, lr_scheduler, self.best_model_save_to)
            self.save_checkpoint(val_accuracy, model, optimizer, lr_scheduler, self.last_checkpoint_save_to,
                                 is_best_model=False) # To replace the last checkpoint as the best model
        elif score <= self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(val_accuracy, model, optimizer, lr_scheduler, self.last_checkpoint_save_to, is_best_model=False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, lr_scheduler, self.best_model_save_to)
            self.save_checkpoint(val_accuracy, model, optimizer, lr_scheduler, self.last_checkpoint_save_to,
                                 is_best_model=False) # To replace the last checkpoint as the best model
            self.counter = 0

        with open(self.pickled, 'wb') as pck:
            pickle.dump({"es_counter": self.counter, "es_best_score": self.best_score}, pck)

    def save_checkpoint(self, val_accuracy, model, optimizer, lr_scheduler, save_to, is_best_model=True):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if is_best_model:
                self.trace_func(f'Parsing accuracy increased ({self.val_accuracy_max:.6f} --> {val_accuracy:.6f}).  Saving model ...')
            else:
                self.trace_func(f'Saving the last checkpoint, parsing accuracy: ({val_accuracy:.6f}).  Saving model ...')
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                    }, save_to)
        self.val_accuracy_max = val_accuracy