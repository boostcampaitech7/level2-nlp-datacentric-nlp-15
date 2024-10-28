import datetime
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

import evaluate
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from model import BERTDataset

# for wandb setting
os.environ['WANDB_DISABLED'] = 'true'
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
#print(DEVICE)

# BASE_DIR is parent directory of src
BASE_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load model
model_name = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

# Load data
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)
#print(f"Train dataset: {len(dataset_train)}")
#print(f"Valid dataset: {len(dataset_valid)}")

data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

def train(run_name : str):
    output_path = os.path.join(MODEL_DIR, run_name)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        eval_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        
        # 수정 안됨
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        num_train_epochs=2,

        # 수정 가능
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )
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

def predict(run_name : str, model):
    output_dir = os.path.join(OUTPUT_DIR, run_name)
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    model.eval()
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

if __name__ == '__main__':
    run_name = input('Please Enter Your Run Name : ')

    while run_name == '':
        run_name = input('Please Enter Your Run Name : ')

    run_name += datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')

    os.makedirs(os.path.join(MODEL_DIR, run_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, run_name), exist_ok=True)

    model = train(run_name)
    predict(run_name, model)