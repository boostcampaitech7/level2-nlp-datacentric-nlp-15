# this file backtranlate a train csv from data folder and do backtranslation
# a model list that we used for backtranslation is down below
# Gemma 8B from huggingface
# LLaMA from huggingface
# MarianMT from huggingface
import os
import pandas as pd
import torch
torch.cuda.empty_cache()

import random
import re
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from konlpy.tag import Okt
from src import data_analysis

parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    okt = Okt()
    # remove "<|END_OF_TURN_TOKEN|>" from text
    before = len(dataset)

    dataset['text'] = dataset['text'].apply(lambda x: x.replace("<|END_OF_TURN|>", ""))
    dataset = dataset[~dataset['text'].str.contains("혼란스러워")]

    # use okt to seperate stem and lemma in last word
    def post_process(text):
        words = okt.pos(text)
        last_word = words[-2][0] if words[-1][1] == 'Punctuation' else words[-1][0]
        last_pos = words[-2][1] if words[-1][1] == 'Punctuation' else words[-1][1]

        if last_pos == 'Verb':
            # remove this last_word from text from backward
            return text.replace(last_word, '', -1)

        elif last_pos == 'Adjective':
            # divide last word into stem and lemma
            stem = okt.morphs(last_word, norm=True, stem=True)[0]
            stem = stem.replace("있다", "")
            stem = stem.replace("하다", "")

            return text.replace(last_word, stem, -1)

        else:
            return text

    dataset['text'] = dataset['text'].apply(post_process)
    dataset['text'] = dataset['text'].apply(lambda x: x.replace(" .", "."))

    after = len(dataset)

    print(f"Before: {before}, After: {after}")
    dataset.to_csv(os.path.join(parent_dir, 'data', 'train_8106_similar_2.csv'), index=False)

    return dataset


def choose_topic(data: pd.DataFrame, n_of_gen=1000, save_every=100) -> pd.DataFrame:
    # get last id num of data 'id' format is like ynat-v1_train_02799
    time_now = pd.Timestamp.now().strftime('%m%d_%H%M%S')
    last_id = data['ID'].iloc[-1]
    last_id = int(last_id.split('_')[-1])

    # topic = list(set) -> set = ('topic_name', topic_count)
    top_topic, last_topic = data_analysis.func5(data)
    topic_dict = {0: '생활문화', 1: '스포츠', 2: '정치', 3: '사회', 4: 'IT과학', 5: '경제', 6: '세계'}
    # shuffle last_topic
    n_label = len(last_topic)
    max_k = 40
    k = 4

    topic_list_t = []
    topic_list_l = []

    model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/aya-expanse-8b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")

    for last_topic_ in last_topic:
        # remove all elem which name length is 1
        topic_list_ = last_topic_.axes[0].tolist()
        topic_list_ = list(filter(lambda x: len(x) > 1, topic_list_))

        topic_list_l.append(topic_list_)

    # for last_topic_ in last_topic:
    #     topic_list_ = last_topic_.axes[0].tolist()[:min(len(last_topic_), max_k)]
    #     topic_list_ = list(filter(lambda x: len(x) > 1, topic_list_))
    #
    #     topic_list_t.append(topic_list_)

    for i in tqdm(range(n_of_gen)):
        idx = i % n_label

        # shuffle and pick top k
        random.shuffle(topic_list_l[idx])
        #random.shuffle(topic_list_t[idx])

        this_topic = topic_list_l[idx][:k]
        #this_topic.extend(topic_list_t[idx][:k])

        random.shuffle(this_topic)

        topic_names = str(this_topic)

        messages = [
            {"role": "user",
             "content": f"다음 주제와 비슷한 주제의 제목을 생성. 10글자 이상의 제목 목록만 출력\"{topic_names}\""},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        outputs = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=196,
            temperature=0.2,
            do_sample=True
        )

        outputs = tokenizer.decode(outputs[0])

        length = len("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
        chatbot_start = outputs.index("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")

        outputs = outputs[chatbot_start + length:]
        outputs = re.sub(r'[^\w\s.]', '', outputs)
        outputs = re.sub(r'END_OF_TURN_TOKEN', '', outputs)
        outputs = outputs.split('\n')[1:]
        outputs.remove('')

        cleaned_array = [re.sub(r'^\d+\.\s*', '', item) for item in outputs]
        random.shuffle(cleaned_array)

        #print(f"\nOriginal Topic - {topic_dict[idx]} : ", this_topic, "-> New Topic : ")

        # append k new data to dataset
        for topic_name in cleaned_array[:min(len(cleaned_array), 2)]:
            last_id += 1
            id_format = f"ynat-v1_train_{last_id}"
            data.loc[len(data)] = {'ID' : id_format, 'text': topic_name, 'target': idx}
            #print(topic_name, end=', ')

        if i % save_every == 0:
            data.to_csv(os.path.join(parent_dir, 'data', f'train_7500_aug_1_{time_now}.csv'), index=False)

    data.to_csv(os.path.join(parent_dir, 'data', f'train_7500_aug_1_{time_now}.csv'), index=False)
    return data

if __name__ == "__main__":
    data_path = os.path.join(parent_dir, 'data', 'train_7500_aug_1.csv')
    dataset = pd.read_csv(data_path)
    data = choose_topic(dataset)
