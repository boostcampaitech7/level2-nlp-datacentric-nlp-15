import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re
from tqdm import tqdm
from huggingface_hub import login

login(token="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")

class ASCIINoiseTextCleaner:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
        
        # ASCII 노이즈 특성
        self.ascii_noise_ranges = [
            (33, 47),   # !"#$%&'()*+,-./
            (58, 64),   # :;<=>?@
            (91, 96),   # [\]^_`
            (123, 126)  # {|}~
        ]
        
    def is_ascii_noise(self, char):
        """ASCII 노이즈 문자인지 확인"""
        if not ord(char) < 128:  # ASCII 범위 확인
            return False
            
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.ascii_noise_ranges)
    
    def get_ascii_noise_ratio(self, text):
        """ASCII 노이즈 비율 계산"""
        noise_chars = sum(1 for char in text if self.is_ascii_noise(char))
        return noise_chars / len(text) if len(text) > 0 else 0
    
    def rule_based_clean(self, text):
        """ASCII 노이즈 특화 규칙 기반 정제"""
        # 1. ASCII 노이즈가 20-80% 사이인 문자들 처리 -> SKIP
        noise_ratio = self.get_ascii_noise_ratio(text)
        
        if 0.9 <= noise_ratio <= 0.95: # 비율을 높임으로써 이 부분은 SKIP 하고 진행
            # 1.1 의미있는 패턴 보존 (제품명, 숫자 등)
            preserved = {}
            patterns = [
                r'[A-Z]+\d*',  # 대문자로 된 제품명
                r'\d+만원',    # 가격
                r'\d+시간',    # 시간
                r'\d+%'       # 비율
            ]
            
            for i, pattern in enumerate(patterns):
                for match in re.finditer(pattern, text):
                    key = f"__PRESERVED_{i}_{match.start()}__"
                    preserved[key] = match.group()
                    text = text.replace(match.group(), key)
            
            # 1.2 ASCII 노이즈 제거
            cleaned = ''
            for char in text:
                if self.is_ascii_noise(char):
                    cleaned += ' '
                else:
                    cleaned += char
            
            # 1.3 보존된 패턴 복원
            for key, value in preserved.items():
                cleaned = cleaned.replace(key, value)
            
            # 1.4 중복 공백 제거
            cleaned = ' '.join(cleaned.split())
            
            return cleaned
        
        return text
    
    def extract_clean_text(self, llm_output):
        """LLM 출력에서 [|assistant|] 태그 다음의 따옴표 안의 텍스트만 추출"""
        if "[|assistant|]" in llm_output:
            # [|assistant|] 이후의 텍스트 추출
            text_after_tag = llm_output.split("[|assistant|]")[1].strip()
            # 첫 번째 따옴표와 그 이후의 첫 번째 따옴표 사이의 텍스트 추출
            match = re.search(r'"([^"]+)"', text_after_tag)
            if match:
                return match.group(1)
        return llm_output  # 패턴이 없으면 원본 반환
    
    def hybrid_clean(self, text):
        """하이브리드 정제 (규칙 기반 + LLM)"""
        # 1. 규칙 기반 정제 먼저 적용
        rule_cleaned = self.rule_based_clean(text)
        noise_ratio = self.get_ascii_noise_ratio(rule_cleaned)
        
        # 2. 여전히 노이즈가 많은 경우 LLM 적용
        if noise_ratio > 0.03:  # 3% 이상의 ASCII 노이즈가 남아있는 경우
            messages = [
    {"role": "system", 
     "content": "당신은 뉴스 기자입니다. 입력받은 문장의 아스키 노이즈를 식별하고 의미적으로 완전한 기사 타이틀을 짧게 생성한 것만 출력합니다."},
    
    {"role": "user", 
     "content": f"의미적으로 완전한 기사 타이틀로 바꿔주세요.: {rule_cleaned}"}
]
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids.to("cuda"),
                    max_new_tokens=64,
                    temperature=0.1,
                    num_beams=1,
                    do_sample=False,
                    early_stopping=True
                    
                )
            
            llm_cleaned = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # 새로운 후처리 함수 적용
            llm_cleaned = self.extract_clean_text(llm_cleaned)
                
            # LLM 결과가 적절한지 검증
            if len(llm_cleaned) >= len(rule_cleaned) * 0.5:
                return llm_cleaned
        
        return rule_cleaned

def main():
    cleaner = ASCIINoiseTextCleaner()
    df = pd.read_csv('/data/ephemeral/home/level2-nlp-datacentric-nlp-15/data/train.csv')
    
    print("텍스트 정제 작업 시작...")
    cleaned_texts = []
    for text in tqdm(df['text'], desc="Processing texts"):
        cleaned_text = cleaner.hybrid_clean(text)
        cleaned_texts.append(cleaned_text)
        
    # 정제된 텍스트로 데이터프레임 업데이트
    df['text'] = cleaned_texts
    
    # 정제된 데이터셋 저장
    output_path = '/data/ephemeral/home/level2-nlp-datacentric-nlp-15/data/train_llm.csv'
    df.to_csv(output_path, index=False)
    print(f"텍스트 정제 작업 완료 ({output_path})")

if __name__ == "__main__":
    main()
    