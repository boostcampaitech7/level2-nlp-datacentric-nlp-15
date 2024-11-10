import os
import pandas as pd
from konlpy.tag import Okt
from collections import defaultdict

import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

parent_dir = os.path.dirname(os.getcwd())
dataset_path = os.path.join(parent_dir, 'data', 'train_8525_15000_cleanlab.csv')

# 1. 데이터 로드 및 전처리
df = pd.read_csv(dataset_path)

# 2. 명사 추출
okt = Okt()
noun_dict = defaultdict(lambda: {'count': 0, 'targets': set(), 'each_target_count': defaultdict(int)})
k_dict = {0 : '0', 1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5', 6 : '6'}
v_dict = {v: k for k, v in k_dict.items()}

def to_text(row):
    noun_dict_ = noun_dict[row['noun']]

    target_count_list = [f'{k}({v})' for k, v in noun_dict_['each_target_count'].items()]
    text = f"{', '.join(target_count_list)}"

    return text

def is_dominating(row, threshold=0.6):
    noun_dict_ = noun_dict[row['noun']]

    maximum_count = max(noun_dict_['each_target_count'].values())
    total_count = sum(noun_dict_['each_target_count'].values())

    return 1 if maximum_count / total_count >= threshold else 0

def get_maximum_count_label_index(row):
    noun_dict_ = noun_dict[row['noun']]

    maximum_count = max(noun_dict_['each_target_count'].values())

    # get key from value
    max_count_key = [k for k, v in noun_dict_['each_target_count'].items() if v == maximum_count][0]

    return v_dict[max_count_key] if row['domination'] == 1 else -1

for _, row in df.iterrows():
    nouns = okt.nouns(row['text'])
    for noun in nouns:
        noun_dict[noun]['count'] += 1
        noun_dict[noun]['targets'].add(k_dict[int(row['target'])])
        noun_dict[noun]['each_target_count'][k_dict[int(row['target'])]] += 1

# for each nouns sort each_target_count by count
for noun in noun_dict:
    noun_dict[noun]['each_target_count'] = dict(sorted(noun_dict[noun]['each_target_count'].items(), key=lambda x: x[1], reverse=True))

# 3. 명사 빈도 및 target별 분포 분석
noun_df = pd.DataFrame([
    {'noun': noun, 'count': data['count'], 'target_count': len(data['targets'])}
    for noun, data in noun_dict.items()
])

# 3.2. 1번만 나오는 명사는 제거
noun_df = noun_df[noun_df['count'] > 1]

# 3.3. 1글자 명사는 제거
noun_df = noun_df[noun_df['noun'].apply(lambda x: len(x) > 1)]
noun_df['domination'] = noun_df.apply(is_dominating, axis=1)
noun_df['target'] = noun_df.apply(get_maximum_count_label_index, axis=1)

def make_ambiguous():
    # 4. 애매한 명사 추출
    threshold_count_min = int(max(noun_df['count'].median(), 3))  # 중앙값을 기준으로 설정
    threshold_count_max = 20  # 중앙값을 기준으로 설정
    threshold_target_min = 3  # 3개 이상의 target에 나타나는 경우를 애매하다고 판단
    threshold_target_max = 6  # 3개 이상의 target에 나타나는 경우를 애매하다고 판단

    ambiguous_nouns = noun_df[
        (noun_df['count'] >= threshold_count_min) &
        (noun_df['count'] < threshold_count_max) &
        (noun_df['target_count'] >= threshold_target_min) &
        (noun_df['target_count'] <= threshold_target_max) &
        (noun_df['domination'] == 0)
    ].sort_values('target_count', ascending=False)

    # 4-2. 애매한 명사에 target_list column 추가
    ambiguous_nouns['target_list'] = ambiguous_nouns.apply(to_text, axis=1)
    ambiguous_nouns = ambiguous_nouns.drop(columns=['domination', 'target'])

    # 4-3. 전체 row를 정렬, 정렬은 1순위로 target_count, 2순위로 count, 3순위로 len(noun)
    ambiguous_nouns = ambiguous_nouns.sort_values(['target_count', 'count', 'noun'], ascending=[False, False, True])
    ambiguous_nouns.to_csv(os.path.join(parent_dir, 'data', 'ambiguous_nouns_8525_1.csv'), index=False)

def make_true():
    # 4. 애매한 명사 추출
    threshold_count_min = 10  # 중앙값을 기준으로 설정
    threshold_target = 5  # 2개 이하 target에 나타나는 경우를 정확하다고 판단

    # 5. 확실한 명사 추출
    true_nouns = noun_df[
        (noun_df['count'] >= threshold_count_min) &
        (noun_df['target_count'] < threshold_target) &
        (noun_df['domination'] == 1)
    ].sort_values('target_count', ascending=False)

    true_nouns['target_list'] = true_nouns.apply(to_text, axis=1)
    true_nouns = true_nouns.drop(columns=['domination', 'target'])

    # 4-3. 전체 row를 정렬, 정렬은 1순위로 target_count, 2순위로 count, 3순위로 len(noun)
    true_nouns = true_nouns.sort_values(['target_count', 'count', 'noun'], ascending=[False, False, True])
    true_nouns.to_csv(os.path.join(parent_dir, 'data', 'true_nouns_8525_1.csv'), index=False)

if __name__ == '__main__':
    make_ambiguous()
    make_true()