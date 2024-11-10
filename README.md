<div align='center'>

  # 🏆 LV.2 NLP 프로젝트 : Open-Domain Question Answering

</div>

## ✏️ 대회 소개

| 특징     | 설명 |
|:------:| --- |
| 대회 주제 | 데이터 중심 자연어처리 대회 - 데이터 품질 개선을 통한 성능 향상 |
| 대회 설명 | 모델 구조 변경 없이 데이터 품질 개선만으로 분류 성능을 향상시키는 대회 |
| 데이터 구성 | 원본 데이터: 2,800개 (노이즈 1,600개, 오라벨링 1,000개 포함) |
| 평가 지표 | Macro F1 Score |


## 🎖️ Leader Board
###  Public Leader Board 


###  Private Leader Board 


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
|개요| 설명 |
|:------:| --- |
| 주제 | 데이터 품질 개선을 통한 뉴스 기사 분류 성능 향상 |
| 목표 | 노이즈 제거, 데이터 증강 등을 통한 F1 Score 개선 |
| 평가 지표 | Macro F1 Score |
| 개발 환경 | Python 3.10, PyTorch, Transformers |
| 협업 환경 | GitHub, Notion, Slack, W&B |

### 📅 프로젝트 타임라인
- 프로젝트는 2024.10.28 - 2024.11.07 (총 11일)

<div align='center'>



</div>

### 🕵️ 프로젝트 진행
- 프로젝트를 진행하며 단계별로 실험하여 적용한 내용들을 아래와 같습니다.


|  프로세스   | 설명 |
|:-------:| --- |
| EDA     | 노이즈/비-노이즈 데이터 분석, 라벨 분포 분석 |
| 전처리   | LLM 기반 노이즈 제거, 명사 기반 중복 데이터 제거 |
| 증강     | LLM을 활용한 데이터 증강, DeepL API를 활용한 역번역 |
| 클러스터링 | GMM, K-Means 활용한 오라벨 데이터 보정, BERT 임베딩 기반 클러스터링 |
| 모델 선정  |  |

### 📊 Dataset
- 데이터 증강 과정에서 라벨 분포를 균형있게 맞추고자 라벨별 증강비율을 조정하였습니다.

|버전| 설명 |크기|
|:-------------------:| --- |---|
|  |  |  |
|  | ` ||

### 🤖 Ensemble Model


| Model | val_pearson | learning_rate| batch_size | 사용 데이터 |
|-------------------------| --- |---|----- |---|


## 📁 프로젝트 구조



### 📦 src 폴더 구조 설명



### 📁 보충 설명


### 📦 Installation

