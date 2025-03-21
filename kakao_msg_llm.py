import os
import zipfile
import pandas as pd
from dotenv import load_dotenv
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import shutil


# 개인정보 저장 방지를 위한 방법 (로깅 비활성화)
openai.api_key = os.getenv("OPEN_API_KEY")

# Optional: API 요청과 응답을 추적하지 않도록 하는 설정
openai.log = None  # OpenAI API 호출에서 로그를 비활성화합니다.

class Keywords(BaseModel):
    description: list = Field(description="AI가 생성한 키워드들")


chatbot_dict_not_use = {"메모어 신청 관련": 0, "슬랙 초대링크 관련": 0, "참가 신청 메시지 관련": 0, "슬랙 이메일 변경": 0,
"모집 기간 관련": 0, "보증금 관련": 0,"멤버쉽/보증금 제도 관련": 0, "멤버쉽/보증금 확인 관련": 0, "회고 및 댓글 관련": 0, "메모어 슬랙 사용 가이드": 0,
"회고/댓글 제출 가이드": 0,  "모임 관련": 0, "모임 편성 관련": 0, "모임 변경 관련":0, "오프라인 모임 장소 관련": 0, "오프라인 및 온라인 비용": 0, "취소/환불 관련": 0, "메모어라이브 관련" : 0,
"신청 확인":0, "참가 조건":0, "메모어아카이빙 관련": 0}

# 모집 기간 관련, 슬랙 이메일 변경, 슬랙 초대링크 관련

chatbot_options = list(chatbot_dict_not_use.keys())

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
parser = JsonOutputParser(pydantic_object=Keywords)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        너는 고객 상담 내용을 분석하는 AI야. 
        주어진 문장에서 핵심적인 키워드를 뽑아줘. 
        키워드는 단어 또는 짧은 구 형태로 구성되어야 해.
        키워드는 중복되지 않도록 해야 하며, 최대 20개까지만 뽑아야 해.
        키워드의 개수는 전체 문장의 50%로 설정해야 해.
        예를 들어, 10줄의 대화가 주어지면 5개의 키워드를 추출해야 해.
        결과는 배열 형식으로 내보내줘.
        또한, 구체적인 날짜 (예: 4월 21일, 2025년 4월 21일)는 키워드에서 제외해야 해.
        날짜를 제외한 나머지 중요한 키워드만 추출해줘.
        추가로, 상투적인 표현인 "도와드렸습니다", "상관없으니", "문의해주세요" 등은 키워드로 포함되지 않도록 제외해줘.
        서비스 이름인 "메모어"도 들어가지 않도록 제외해줘.
        """),
       ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

def get_keyword(query):

    # 모델 초기화
    # temperature이 0에 가까울 수록 정확도가 높은 답변만을 선택함 (1에 가까울수록 랜덤성을 가미함)
    model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", api_key=api_key)

    chain = prompt | model | parser

    response = chain.invoke({"question": query}) 

    return response


def process_csv(file_path):
    # CSV 파일 읽기
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    df = pd.DataFrame(df)

    # "USER"이 "memoir(메뉴)"인 메시지와 "MESSAGE" 내용이 chatbot_options에 포함된 메시지 제외
    # 또한 "USER"이 "memoir"이고 "MESSAGE"가 "[메모어 XX기 X주차]" 형식으로 시작하는 메시지 제외
    filtered_df = df[
        ~(df["USER"] == "memoir(메뉴)") &  # "USER"이 "memoir(메뉴)"인 메시지 제외
        ~df["MESSAGE"].isin(chatbot_options) &  # "MESSAGE"가 chatbot_options에 있는 메시지 제외
        ~((df["USER"] == "memoir") & 
          (df["MESSAGE"].str.match(r"^\[메모어 \d+기 \d+주차\]")))]  # "[메모어 XX기 X주차]" 형식 제외


    # 필터링된 메시지들 추출
    user_messages = filtered_df["MESSAGE"]

    # 메시지를 "\n"으로 연결하여 하나의 문자열로 변환
    message_string = "\n".join(user_messages)

    # AI 모델을 통해 키워드 추출
    keyword_result = get_keyword(message_string)

    print(">>>>>>>>>>>>> ✅ 키워드 추출 성공 <<<<<<<<<<<<<<")
    return keyword_result

# print(process_csv("memoir_조혜미.csv"))

def process_files(folder):
    # 폴더 내 CSV 파일 목록 가져오기
    csv_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    print(f"CSV 파일 목록: {csv_files}")  # 확인용
    
    # 각 CSV 파일에 대해 키워드 추출
    all_keywords = {}
    
    for file in csv_files:
        print(f"Processing file: {file}")
        keywords = process_csv(file)
        all_keywords[file] = keywords
    
    return all_keywords

# folder = 'memoir_last_10'  # CSV 파일들이 저장된 폴더 경로 지정
# result = process_files(folder)

# # 결과 출력
# # print(result)
# # # 결과 출력
# for file, keywords in result.items():
#     print(f"파일: {file}")
#     print(f"추출된 키워드: {keywords}")

# process_csv("memoir_조혜미.csv")
import spacy
from typing import Dict

# 가능한 카테고리 리스트 정의
ALLOWED_CATEGORIES = ['서비스', '회고', '모임', '보증금', '신청', '기타']

# 예시: 각 카테고리별로 관련된 키워드들을 정의
CATEGORY_KEYWORDS = {
    "서비스": [
        "고객", "지원", "서비스", "서비스팀", "상담", "운영", 
        "서비스제공", "고객지원", "고객센터", "해결", "서비스이용", "이용안내",
        "문제해결", "피드백", "클레임", "서비스품질", "고객경험", "불만", "문의"
    ],
    "회고": ["회고", "반성", "분석", "성찰", "교훈"],
    "모임": [
        "모임", "모임장", "만남", "회식", "모임일정", "모임공고", 
        "모임시간", "모임장소", "모임참석", "모임초대", "모임목적", "회합", 
        "단체활동", "팀워크", "네트워킹", "이벤트", "워크숍", "세미나"
    ],
    "보증금": ["보증금", "deposit", "보증", "보증인", "렌탈", "환불"],
    "신청": ["신청", "지원", "등록", "신청서", "가입"],
    "기타": []
}

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_lg")

# 키워드를 카테고리로 분류하는 함수
def categorize_keywords_with_spacy(keywords):
    # 결과를 담을 딕셔너리
    category_dict = {category: [] for category in ALLOWED_CATEGORIES}

    # 각 키워드를 분석하여 카테고리로 분류
    for keyword in keywords:
        matched_category = None
        keyword_doc = nlp(keyword)  # spaCy로 키워드 분석

        # 각 카테고리별 키워드와 비교
        for category, category_keywords in CATEGORY_KEYWORDS.items():
            for cat_keyword in category_keywords:
                cat_doc = nlp(cat_keyword)
                similarity = keyword_doc.similarity(cat_doc)  # 의미 유사도 계산

                if similarity > 0.7:  # 유사도가 0.7 이상일 경우 매칭
                    matched_category = category
                    break

            if matched_category:
                break

        if matched_category is None:  # 해당 카테고리와 일치하지 않으면 '기타'로 처리
            matched_category = "기타"
        
        category_dict[matched_category].append(keyword)

    return category_dict

def chatbot_category(file_path, chatbot_dict):
    # CSV 파일 읽기
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    df = pd.DataFrame(df)

    for message in df["MESSAGE"]:
        if message in chatbot_options:
            chatbot_dict[message] += 1


    return chatbot_dict
