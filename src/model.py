import torch
from torch.utils.data import Dataset
from arguments import DataTrainingArguments

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, data_args: DataTrainingArguments):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            # 수정 가능
            tokenized_input = tokenizer(
                text, 
                padding='max_length' if data_args.pad_to_max_length else False, 
                truncation=True, 
                return_tensors='pt'
            )
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)