from torch.nn.utils.rnn import pad_sequence

class Collator(object):
    def __init__(self, device, pad_labels=True):

        # whether to pad or not the labels
        # if loss needs to be computed => True
        # if only smatch score, blue score => False

        self.device = device
        self.pad_labels = pad_labels


    def __call__(self, batch):

        batch = [{key: encoding[key].to(self.device, non_blocking=True) for key in encoding} for encoding in batch]
        input_ids = [encoding['input_ids'][0, :] for encoding in batch]
        attention_mask = [encoding['attention_mask'][0, :] for encoding in batch]
        labels = [encoding['labels'][0, :] for encoding in batch]

        input_ids_padded = pad_sequence(input_ids, padding_value=1, batch_first=True)  # padded_token = <pad>
        attention_mask_padded = pad_sequence(attention_mask, padding_value=0, batch_first=True)  # attention_mask = 0

        if self.pad_labels:
            # padding value for labels = -100 (see hf doc for mbart)
            labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        padded_batch = {'input_ids': input_ids_padded,
                        'attention_mask': attention_mask_padded,
                        'labels': labels}

        return padded_batch