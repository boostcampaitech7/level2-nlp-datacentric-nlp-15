import os
import os
# open csv file as pandas dataframe
import pandas as pd

# open csv file as pandas dataframe
import pandas as pd

import matplotlib.pyplot as plt
from konlpy.tag import Okt
from collections import defaultdict

def func1():
    # save csv as utf-8
    new_data_path = os.path.join(parent_dir, 'data', 'train_utf16.csv')
    train_data.to_csv(new_data_path, index=False, encoding='utf-16')

    input_texts = train_data['text']
    random_10 = input_texts.sample(10)
    print(random_10)

def func2():
    ten_data = test_data.sample(10)
    new_data_path = os.path.join(parent_dir, 'data', 'test_short.csv')
    ten_data.to_csv(new_data_path, index=False, encoding='utf-8')

def func3(train_data : pd.DataFrame):
    random_sample_200 = 200

    # random sample 200 data rows
    random_sample = train_data.sample(random_sample_200)
    random_sample.to_csv(os.path.join(parent_dir, 'data', 'train_sample_200.csv'), index=False)

def func4(train_data : pd.DataFrame):
    # this function reads train csv text and show how many korean characters are in the text
    import re
    korean_char = re.compile('[가-힣]')
    train_data['korean_char_count'] = train_data['text'].apply(lambda x: len(korean_char.findall(x)))

    # show in pandas dataframe bar plot
    train_data['korean_char_count'].plot(kind='hist', bins=50, title='Korean Character Count in Text')

    import matplotlib.pyplot as plt
    plt.show()

    print(train_data['korean_char_count'].describe())

def func5(train_data : pd.DataFrame):
    label_counts = train_data['target'].value_counts().sort_index()
    label_counts.plot(kind='bar', title='Label Distribution')

    # get top-K nouns for each target label and show in pandas dataframe
    text = ""
    okt = Okt()
    top_k = 50
    n_labels = len(label_counts)
    label_nouns_list, least_label_nouns_list, = [], []

    def extract_nouns(text):
        return okt.nouns(text) if isinstance(text, str) else []

    for label in range(n_labels):
        label_data = train_data[train_data['target'] == label].copy()
        label_data.loc[:, 'nouns'] = label_data['text'].apply(extract_nouns)

        label_nouns = label_data['nouns'].sum()
        label_nouns = pd.Series(label_nouns)
        label_nouns, least_label_nouns = label_nouns.value_counts(), label_nouns.value_counts().sort_values(ascending=True)

        # text += f"Label {label} Top {top_k} Nouns\n"
        # text += str(label_nouns[:top_k]) + '\n\n'
        # text += f"Label {label} Least {top_k} Nouns\n"
        # text += str(least_label_nouns[:top_k]) + '\n\n'

        label_nouns_list.append(label_nouns)
        least_label_nouns_list.append(least_label_nouns)

    #plt.show()
    # write txt file
    # with open(os.path.join(parent_dir, 'data', 'label_nouns.txt'), 'w', encoding='utf-8') as f:
    #     f.write(text)

    return label_nouns_list, least_label_nouns_list


def func6(train_data : pd.DataFrame):
    okt = Okt()
    label_nouns = defaultdict(set)
    noun_dict = dict()

    # 각 텍스트에서 명사 추출 및 라벨별로 저장
    for _, row in train_data.iterrows():
        text = row['text']
        label = row['target']
        nouns = set(okt.nouns(text))  # 중복 제거를 위해 set 사용
        label_nouns[label].update(nouns)

        for noun in nouns:
            if noun in noun_dict:
                noun_dict[noun] += 1
            else:
                noun_dict[noun] = 1

    # 2개 이상의 라벨에서 공통으로 발견된 명사 찾기
    common_nouns = defaultdict(list)
    all_nouns = set()

    for nouns in label_nouns.values():
        all_nouns.update(nouns)

    for noun in all_nouns:
        c_noun = sum(1 for label_set in label_nouns.values() if noun in label_set)
        common_nouns[c_noun].append([noun, noun_dict[noun]])

    text = ""
    # 결과 출력
    for i in range(2, len(label_nouns) + 1):
        text += f"라벨 {i}개에서 공통으로 발견된 명사\n"
        text += ''.join(str(sorted(common_nouns[i], key=lambda x:-x[-1]))) + "\n\n"

    with open(os.path.join(parent_dir, 'data', 'common_nouns.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    data_path_train = os.path.join(parent_dir, 'data', 'train.csv')
    data_path_test = os.path.join(parent_dir, 'data', 'test.csv')

    train_data = pd.read_csv(data_path_train)
    test_data = pd.read_csv(data_path_test)

    #func2()
    #func3()
    #func4()
    func5(train_data)
    #func6()