import os
import transformers
import datasets
import torch
import pandas as pd
import re
from nltk.corpus import words

import nltk
nltk.download('words')

english_words = set(words.words())
def remove_gibberish(text):
    english_words_in_text = re.findall(r'[A-Za-z]+', text)
    gibberish_words = [word for word in english_words_in_text if word.lower() not in english_words]
    for word in gibberish_words:
        text = text.replace(word, '')
    return text

from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import random
import numpy as np

# SEED = 2024
# set_seed(SEED)
parent_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(parent_dir, 'data', 'train.csv')

train_data = pd.read_csv(data_path)

# remove r"[^\w\s.]" from text using re sub
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r"[^\w\s.]", "", x))
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'[①-⑩]', "", x))

# remove r'(?<=[가-힣])[a-zA-Z](?=[가-힣])' from text using re sub
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'(?<=[가-힣])[a-zA-Z](?=[가-힣])', "", x))

# remove r'([가-힣])[a-zA-Z] '
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'([가-힣])[a-zA-Z] ', r'\1', x))

# remove r' [a-zA-Z]([가-힣])'
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r' [a-zA-Z]([가-힣])', r' \1', x))

# remove number-english-number
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'\d+[a-zA-Z]+\d+', '', x))

# remove all Gibberish characters using nltk words
train_data['text'] = train_data['text'].apply(lambda x: remove_gibberish(x))

# remove all single alphabet characters
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'(?<![a-zA-Z])[b-zB-Z](?![a-zA-Z])', '', x))

# remove small alphabet and large alphabet after
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'(?<![a-zA-Z])[a-z]+[A-Z][a-zA-Z]*(?![a-zA-Z])', '', x))

# remove eng word that repeats same alphabet more than 2 times
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'\b\w([a-zA-Z])\1+\b', '', x))

# drop all the rows that has less then 7 korean characters
korean_char = re.compile('[가-힣]')
train_data['korean_char_count'] = train_data['text'].apply(lambda x: len(korean_char.findall(x)))
train_data = train_data[train_data['korean_char_count'] > 7]

# save csv as utf-8
time_now = pd.Timestamp.now().strftime("%d_%H%M%S")
new_data_path = os.path.join(parent_dir, 'data', f'train_eda_{time_now}.csv')

train_data.to_csv(new_data_path, index=False)