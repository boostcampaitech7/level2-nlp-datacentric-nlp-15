# this py returns backtranslation of the input (train.csv)
# a model using for backtranslation is provided in the huggingface model hub up to 7B parameters
# translation process should be ko -> xx -> ko

import os
import transformers
import datasets
import torch
import pandas as pd

from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed: int = 456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# SEED = 2024
# set_seed(SEED)
parent_dir = os.path.dirname(os.getcwd())

train_data_name = "train_sample_200_trim_ver_0.csv"
src_lang = "kor_Hang"
dst_lang = "eng_Latn"
model_name = "facebook/nllb-200-3.3B"
model_name_short = model_name.split("/")[-1].replace("-", "_")
def forward_translation():
    data_path = os.path.join(parent_dir, 'data', train_data_name)
    data = pd.read_csv(data_path)

    forward_tokenizer = AutoTokenizer.from_pretrained(model_name)
    forward_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    forward_tokenizer.src_lang = src_lang

    # Random Sample 200 data
    input_texts = data['text'][:200].tolist()

    # Translate the input texts
    translated_texts = []

    for text in tqdm(input_texts):
        # Translate from Korean to English
        forward_input = forward_tokenizer(text, return_tensors="pt", padding=True)
        forward_output = forward_model.generate(
            **forward_input,
            forced_bos_token_id=forward_tokenizer.lang_code_to_id[dst_lang]
        )
        forward_translation = forward_tokenizer.batch_decode(forward_output, skip_special_tokens=True)

        translated_texts.append(forward_translation)

    torch.cuda.empty_cache()

    # Append original texts
    data['trans_text'] = translated_texts
    data.to_csv(os.path.join(parent_dir, 'data', f'train_translated_en_{model_name_short}_2.csv'), index=False)

# backtranslation
def back_translation():
    data_path = os.path.join(parent_dir, 'data', f'train_translated_en_{model_name_short}.csv')
    data = pd.read_csv(data_path)

    input_texts = data['trans_text'].tolist()
    translated_texts = []

    backward_tokenizer = AutoTokenizer.from_pretrained(model_name)
    backward_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    backward_tokenizer.src_lang = dst_lang

    for text in tqdm(input_texts):
        # Translate from English to Korean
        backward_input = backward_tokenizer(text, return_tensors="pt", padding=True)
        backward_output = backward_model.generate(
            **backward_input,
            forced_bos_token_id=backward_tokenizer.lang_code_to_id[src_lang]
        )
        backward_translation = backward_tokenizer.batch_decode(backward_output, skip_special_tokens=True)

        translated_texts.append(backward_translation)

    torch.cuda.empty_cache()

    # concat the translated texts with data pd
    data['back_translated_text'] = translated_texts
    data.to_csv(os.path.join(parent_dir, 'data', f'train_translated_en_{model_name_short}.csv'), index=False)


if __name__ == '__main__':
    forward_translation()
    #back_translation()