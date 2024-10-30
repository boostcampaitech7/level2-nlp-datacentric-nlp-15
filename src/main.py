import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import sys

import torch

import evaluate
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser

from sklearn.model_selection import train_test_split
from model import BERTDataset
from arguments import ModelArguments, DataTrainingArguments

os.environ["WANDB_MODE"] = "online"
logger = logging.getLogger(__name__)

def set_seed(seed: int = 456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

SEED = 456
set_seed(SEED)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main(run_name):
    parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments)
        )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    project_prefix = "[train]" if train_args.do_train else "[eval]" if train_args.do_eval else "[pred]"
    wandb.init(
        project="data_centric",
        entity="nlp15",
        name=f"{project_prefix}_{run_name}",
        save_code=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"model is from {model_args.model_name_or_path}")
    logging.info(f"data is from {data_args.dataset_name}")

    # Load model
    if train_args.do_train:
        model_name = model_args.model_name_or_path #'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not train_args.do_train:
        model_name = os.path.join(model_args.model_name_or_path, 'checkpoint-124')
        tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    # Load data
    data = pd.read_csv(data_args.dataset_name) #pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    dataset_train, dataset_valid = train_test_split(data, test_size=model_args.train_test_split, random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer, data_args)
    data_valid = BERTDataset(dataset_valid, tokenizer, data_args)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if train_args.do_train:
        model = train(model, data_train, data_valid, data_collator, run_name, train_args, data_args)
    if train_args.do_predict:
        predict(model, tokenizer, train_args, data_args)

def train(model, data_train, data_valid, data_collator, run_name : str, train_args: TrainingArguments, data_args: DataTrainingArguments):
    # output_path = os.path.join(MODEL_DIR, run_name)

    training_args = TrainingArguments(
        output_dir=train_args.output_dir, #output_path,
        overwrite_output_dir=True,
        do_train=train_args.do_train,
        do_eval=train_args.do_eval,
        do_predict=train_args.do_predict,
        logging_strategy='steps',
        eval_strategy='steps',
        save_strategy='steps',
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        
        # 수정 안됨
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        num_train_epochs=2,

        # 수정 가능
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )

    f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average='macro')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model

def predict(model, tokenizer, train_args: TrainingArguments, data_args: DataTrainingArguments = None, run_name : str = None):
    dataset_test = pd.read_csv(data_args.test_dataset_name) #pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    model.eval()
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    dataset_test['target'] = preds

    os.makedirs(train_args.output_dir, exist_ok=True)
    dataset_test.to_csv(os.path.join(train_args.output_dir, 'predictions.csv'), index=False)

if __name__ == '__main__':
    try:
        argv_run_index = sys.argv.index('--run_name') + 1
        argv_run_name = sys.argv[argv_run_index]

    except ValueError:
        argv_run_name = ''
        while argv_run_name == '':
            argv_run_name = input("run name is missing, please add run name : ")

    main(argv_run_name)