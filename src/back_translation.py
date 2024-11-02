# this file backtranlate a train csv from data folder and do backtranslation
# a model list that we used for backtranslation is down below
# Gemma 8B from huggingface
# LLaMA from huggingface
# MarianMT from huggingface
import os
import pandas as pd
import torch
from tqdm import tqdm

# pip install bitsandbytes accelerate
from transformers import AutoConfig, AutoModel, AutoTokenizer
torch.cuda.empty_cache()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

# Choose your prompt
prompt = "Explain who you are"  # English example
#prompt = "너의 소원을 말해봐"   # Korean example

messages = [
    {"role": "system",
     "content": "You are EXAONE model from LG AI Research, a helpful assistant."},

    {"role": "user",
     "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to("cuda"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128
)

print(tokenizer.decode(output[0]))


# for i, elem in tqdm(enumerate(dataset['text'])):
#     input_text = prompt + "\"" + elem + "\""
#     input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
#
#     outputs = model.generate(**input_ids, max_new_tokens=32)
#     outputs = tokenizer.decode(outputs[0])
#
#     dataset.loc[i, 'eng'] = outputs
#
# dataset.to_csv(os.path.join(parent_dir, 'data', 'train_8130_eng.csv'), index=False)