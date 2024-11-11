<div align='center'>

  # 🏆 LV.2 NLP 프로젝트 : '주제 분류 프로젝트'

</div>

## ✏️ 대회 소개
<div align='center'>

| 특징     | 설명 |
|:------:| --- |
| 대회 주제 | 데이터 중심 자연어처리 대회 - 데이터 품질 개선을 통한 성능 향상 |
| 대회 설명 | 모델 구조 변경 없이 데이터 품질 개선만으로 분류 성능을 향상시키는 대회 |
| 데이터 구성 | 원본 데이터: 2,800개 (노이즈 1,600개, 오라벨링 1,000개, 정상 데이터 200개) |
| 평가 지표 | Macro F1 Score |

</div>

## 🎖️ Leader Board
### 🥇 Public Leader Board (1위)
![image](https://github.com/user-attachments/assets/77b4b9df-6a6e-4cbc-a2c4-2acc1fdcb78b)


### 🥇 Private Leader Board (1위)
![image](https://github.com/user-attachments/assets/fa5d5d30-8ed0-4755-bc41-6ad0b72d751c)


## 👨‍💻 15조가십오조 멤버
<div align='center'>
  
|김진재 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/jin-jae)| 박규태 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/doraemon500)|윤선웅 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/ssunbear)|이정민 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/simigami)|임한택 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/LHANTAEK)|
|:-:|:-:|:-:|:-:|:-:|
|<img src='https://avatars.githubusercontent.com/u/97018331' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/64678476' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/117508164' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/46891822' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/143519383' height=125 width=125></img>|

</div>

  
## 👼 역할 분담
<div align='center'>

|팀원   | 역할 |
|------| --- |
| 김진재 |Clustering​, Data Augmentation​, LLM Relabeling  |
| 박규태 |EDA​, Data Denoise​, Data Augmentation​, LLM Relabeling​  |
| 윤선웅 |Data Augmentation​, Clustering​, LLM Relabeling |
| 이정민 |Data Denoise​, Noun Removal​, Clustering​  |
| 임한택 |Data Denoise​, Back Translation​, Data Augmentation  |
</div>


## 🏃 프로젝트
### 🖥️ 프로젝트 개요
<div align='center'>

|개요| 설명                                 |
|:------:|------------------------------------|
| 주제 | 데이터 품질 개선을 통한 뉴스 기사 주제 분류 성능 향상 |
| 목표 | 노이즈 제거, 데이터 증강 등을 통한 F1 Score 개선 |
| 평가 지표 | Macro F1 Score                     |
| 개발 환경 | Python 3.10, PyTorch, Transformers |
| 협업 환경 | GitHub, Notion, Slack, W&B         |
</div>

### 📅 프로젝트 타임라인
- 프로젝트는 2024.10.28 - 2024.11.07 (총 11일)
![image](https://github.com/user-attachments/assets/4e017529-dbc7-4cd3-80ca-b8ca23c2296d)

<div align='center'>
</div>

### 🕵️ 프로젝트 진행
- 프로젝트를 진행하며 단계별로 실험하여 적용한 내용들을 아래와 같습니다.
<div align='center'>

|  프로세스   | 설명 |
|:-------:| --- |
| EDA     | 노이즈/비-노이즈 데이터 분석, 라벨 분포 분석 |
| 전처리   | LLM 기반 노이즈 제거, 명사 기반 중복 데이터 제거 |
| 증강     | LLM을 활용한 데이터 증강, DeepL API를 활용한 역번역 |
| 클러스터링 | GMM, K-Means 활용한 오라벨 데이터 보정 |
| LLM 선정  | 1. EXAONE-3.0-7.8B-Instruct (KoMT 벤치마크 고성능)<br>2. aya-expanse-8b (m-ArenaHard 벤치마크 고성능)<br>3. ko-gemma-2-9b-it (Horangi Leaderboard 고성능) |

</div>

### 📊 Dataset
- 데이터 증강 및 정제 과정을 통해 원본 데이터에서 고품질의 최종 데이터셋을 구축했습니다.
<div align='center'>

|단계| 설명 |크기|
|:-------------------:| --- |---|
| Raw Data | 원본 데이터 (노이즈 1,600개 포함) | 2,800개 |
| Final Dataset | LLM 기반 노이즈 제거, 데이터 증강, 역번역, 클러스터링 보정을 통해 구축된 최종 데이터 | 15,780개 |
</div>


## 📁 프로젝트 구조
프로젝트 폴더 구조는 다음과 같습니다.
```
level2-datacentric-nlp-15
├── data
│   ├── test_dataset
│   └── train_dataset
├── models
├── output
├── README.md
├── requirements.txt
├── run.py
└── src
    ├──arguments.py
    ├──main.py
    ├──model.py
    ├─back_translation
    │ └── back_translation.ipynb
    │
    ├─clustering
    │ └── clustering.ipynb
    │
    ├─LLM_noise_tasks
    │ ├── char_filter.py
    │ ├── LLM_aug.ipynb
    │ ├── LLM_cleaning_noise.ipynb
    │ └── LLM_label_filtering.ipynb
    │
    ├─noun_removal
    │ └── noun_analysis.py
    │
    └─post_processing_cleanlab
      └── post_processing.ipynb
```


## 📦 src 폴더 구조 설명
```
• arguments.py : 데이터 증강을 하는 파일
• main.py : 모델 train, eval, prediction 을 수행하는 파일
• model.py : 입력 텍스트와 레이블 데이터를 BERT 모델 학습에 맞게 토크나이즈하고 텐서 형식으로 변환해주는 PyTorch Dataset 구현한 파일
• back_translation.ipynb : 역번역 테스크을 수행하는 파일 
• clustering.ipynb : 클러스터링 테스크를 수행하는 파일
• char_filter.py : 불필요한 noise 데이터를 전/후 처리하는 파일
• LLM_aug.ipynb : LLM 을 활용한 데이터 생성, 증강을 수행하는 파일
• LLM_cleaning_noise.ipynb : LLM 을 활용한 noise를 판별하고 denoise을 수행하는 파일
• LLM_label_filtering.ipynb : LLM 을 활용해서 텍스트들의 주제를 뽑고 재-라벨링, 생성을 수행하는 파일
• noun_analysis.py : 단어의 빈도를 분석하여서 데이터를 정제하여 개선을 수행하는 파일
• post_processing.ipynb : cleanlab 을 활용하여서 데이터를 정제, 제거를 수행하는 파일
```


## 📦 Installation
```
- python=3.10 환경에서 requirements.txt를 pip로 install 합니다. (pip install -r requirements.txt)
- python run.py를 입력하여 프로그램을 실행합니다.
```
