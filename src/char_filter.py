import re
import pandas as pd

def filter_sentences(sentences):
    english_pattern = re.compile(r'^[a-zA-Z\s\.,!?\'\"\n\-]+$')
    hanja_pattern = re.compile(r'^[\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\s\.,!?\'\"\n\-]+$')
    
    if english_pattern.fullmatch(sentences):
        print(sentences)
        return True
    elif hanja_pattern.fullmatch(sentences):
        print(sentences)
        return True
    elif '\n' in sentences:
        print(sentences)
        return True
    else:
        return False

df = pd.read_csv("../data/nanoised_filtered_aug.csv")

datas = []
for i, row in df.iterrows():
    result = filter_sentences(row['text'])
    if result:
        continue
    else:
        text = row['text'].strip("*\"'+")
        datas.append({
            'ID': row['ID'],
            'text': text,
            'target': row['target']
        })

filtered_df = pd.DataFrame(datas)
filtered_df.to_csv("../data/nanoised_filtered_aug2.csv", index=False)