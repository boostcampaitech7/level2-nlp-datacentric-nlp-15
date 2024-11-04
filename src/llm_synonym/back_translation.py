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

parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)

def exaone_synonym(dataset):
    model = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

    prompt = ""

    messages = [
        {"role": "system",
         "content": "명사와 동사를 이용하여 비슷한 한국어 문장을 설명 없이 생성."},

        {"role": "user",
         "content": "다저스 _버츠 감독J류현4 담당 포수!최! 환^ 만들P...특별법 흔들"},

        {"role": "assistant",
         "content": "번역 : 다저스 버츠 감독이 류현진과 담당 포수 최환을 만듬. 특별법 흔들"},

        {"role": "user",
         "content": "번역 : 월드9U천재 사령탑 S시코 !'O오 감독 한국전R..."},

        {"role": "assistant",
         "content": "번역 : 세계적 천재의 사령탑 시코! 오 감독과 한국전 치루다."},

        {"role": "user",
         "content": "HG8플러스 Nx^H에5J5G htW ~스트 성공"},

        {"role": "assistant",
         "content": "번역 : HG8 플러스 NH에 5대5로 호스트 성공!"},

        {"role": "user",
         "content": "완전한!…충북 F1기 8*1R∼EgI 큰 일 주의"},

        {"role": "assistant",
         "content": "번역 : 완전한 충북 1기, 8대 1로 큰 일 주의!"},

        {"role": "user",
         "content": prompt},

    ]

    for i, elem in tqdm(enumerate(dataset['text'])):
        # input_text = prompt + "\"" + elem + "\""
        # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

        messages[-1]["content"] = elem

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        outputs = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=128
        )

        outputs = tokenizer.decode(outputs[0])

        # outputs = model.generate(**input_ids, max_new_tokens=32)
        # outputs = tokenizer.decode(outputs[0])

        if "해석하기 어렵습니다" in outputs or "파악하기 어렵습니다" in outputs:
            # drop this row from dataset
            dataset.drop(i, inplace=True)

        else:
            # split outputs by enter
            outputs = outputs.split('\n')
            outputs = list(filter(None, outputs))

            outputs = outputs[len(messages):]

            for elem_ in outputs:
                if "번역 : " in elem_:
                    changed = elem_.replace("[|assistant|]", "")
                    changed = changed.replace("[|user|]", "")
                    changed = changed.replace("[|endofturn|]", "")
                    changed = changed.replace("번역 : ", "")
                    changed = changed.replace("\"", "")

                    print(f"B: {elem} -> A: {changed}")
                    dataset.loc[i, 'text'] = changed
                    break

    dataset.to_csv(os.path.join(parent_dir, 'data', 'train_8130_similar.csv'), index=False)

def aya_synonym(dataset):
    model = AutoModelForCausalLM.from_pretrained(
        "CohereForAI/aya-expanse-8b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")

    prompt = ""
    #prompt = "인공지능의 발전은 진보인가"
    #prompt = "월드9U천재 사령탑 S시코 !'O오 감독 한국전R..."
    #prompt = "A일!리c 산 _R법정=l 내 처리요"

    messages = [
        {"role": "user",
         "content": f"다음 문장을 명사와 동사를 바꾼 비슷한 문장 1개로 바꿔줘 : \"{prompt}\""},
    ]

    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # )
    # outputs = model.generate(
    #     input_ids.to("cuda"),
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=128,
    #     temperature=0.2,
    #     do_sample=True
    # )
    #
    # outputs = tokenizer.decode(outputs[0])
    # print(outputs)

    for i, elem in tqdm(enumerate(dataset['text'])):
        prompt = elem
        messages = [
            {"role": "user",
             "content": f"다음 문장을 명사와 동사를 바꾼 비슷한 문장 1개로 바꿔줘 : \"{prompt}\""},
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
        outputs = outputs[chatbot_start+length:]

        # fine \" index and string between them
        start = outputs.find("\"")
        end = outputs.find("\"", start + 1)

        if start == -1 or end == -1:
            if "**변경된 문장:** " in outputs:
                sentences = outputs.split('\n\n')
                changed = sentences[0].replace("**변경된 문장:** ", "")
                print(f"B: {elem} -> A: {changed}")
                dataset.loc[i, 'text'] = changed

            else:
                continue

        else:
            changed = outputs[start + 1:end]
            print(f"B: {elem} -> A: {changed}")
            dataset.loc[i, 'text'] = changed

    dataset.to_csv(os.path.join(parent_dir, 'data', 'train_8250_similar.csv'), index=False)


if __name__ == "__main__":
    data_path = os.path.join(parent_dir, 'data', 'train_clustered_8130.csv')
    dataset = pd.read_csv(data_path)

    #exaone_synonym(dataset)
    data_path = os.path.join(parent_dir, 'data', 'train_clustered_8250.csv')
    dataset = pd.read_csv(data_path)

    aya_synonym(dataset)