import os
import zipfile
import pandas as pd
from dotenv import load_dotenv
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model
from enum import Enum
from typing import List, Set
import shutil


# 개인정보 저장 방지를 위한 방법 (로깅 비활성화)
openai.api_key = os.getenv("OPEN_API_KEY")

# Optional: API 요청과 응답을 추적하지 않도록 하는 설정
openai.log = None  # OpenAI API 호출에서 로그를 비활성화합니다.


chatbot_dict_not_use = {"메모어 신청 관련": 0, "슬랙 초대링크 관련": 0, "참가 신청 메시지 관련": 0, "슬랙 이메일 변경": 0,
"모집 기간 관련": 0, "보증금 관련": 0,"멤버쉽/보증금 제도 관련": 0, "멤버쉽/보증금 확인 관련": 0, "회고 및 댓글 작성 관련": 0, "메모어 슬랙 사용 가이드": 0,
"회고/댓글 제출 가이드": 0,  "모임 관련": 0, "모임 편성 관련": 0, "모임 변경 관련":0, "오프라인 모임 장소 관련": 0, "오프라인 및 온라인 비용": 0, "취소/환불 관련": 0, "메모어라이브 관련" : 0,
"신청 확인":0, "참가 조건":0, "메모어아카이빙 관련": 0}



# 모집 기간 관련, 슬랙 이메일 변경, 슬랙 초대링크 관련

chatbot_options = list(chatbot_dict_not_use.keys())
chatbot_large_options = ["메모어 신청 관련", "보증금 관련", "회고 및 댓글 작성 관련", "모임 관련", "메모어라이브 관련", "취소/환불 관련", "메모어아카이빙 관련"]

CategoryEnum = Enum("CategoryEnum", {opt: opt for opt in chatbot_options})
# 중복을 제거하려면 `List`를 사용하고, 후처리로 중복 제거
class AICategory(BaseModel):
    category: List[CategoryEnum]

# JSON 파서를 설정
parser = JsonOutputParser(pydantic_object=AICategory)

# 중복 제거 로직 추가
def remove_duplicates(categories: List[str]) -> List[str]:
    return list(set(categories))  # set을 이용해 중복 제거

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
        너는 고객 상담 내용을 분석해 해당 내용이 어떤 범주에 속하는지 분류하는 AI야. 
        하나의 대화에서 여러 범주가 나올 수도 있어. 하지만, 각 범주는 하나만 선택되어야 하며 중복되지 않도록 분류해야 해.
        
        범주는 반드시 {', '.join(chatbot_options)} 중 하나 이상이어야 하며, 
        각 범주의 기준은 다음과 같아:

        1. **메모어 신청 관련**:  
           - 메모어 참가 신청, 신청 방법, 얼리버드 신청 관련 문의  
           - 신청 후 절차 안내 요청  

        2. **슬랙 초대링크 관련**:  
           - 슬랙 초대 링크가 만료됨  
           - 초대 링크를 받지 못했거나 다시 요청하는 경우  

        3. **참가 신청 메시지 관련**:  
           - 참가 신청 후 신청 여부 확인  
           - 신청이 정상적으로 접수되었는지 확인 요청  

        4. **슬랙 이메일 변경**:  
           - 슬랙에 등록된 이메일을 변경하고 싶다는 요청  

        5. **모집 기간 관련**:  
           - 모집 일정, 신청 마감일 관련 질문  
           - 모집이 끝났는지 여부 확인  

        6. **보증금 관련**:  
           - 보증금 금액, 납부 방법, 입금 확인 요청  
           - 보증금 환불 관련 문의  

        7. **멤버쉽/보증금 제도 관련**:  
           - 멤버십 및 보증금 제도의 구조에 대한 설명 요청  
           - 보증금이 어떻게 운영되는지 궁금해하는 경우  

        8. **멤버쉽/보증금 확인 관련**:  
           - 보증금이 정상적으로 납부되었는지 확인 요청  
           - 보증금 환불 진행 여부 확인  

        9. **회고 및 댓글 작성 관련**:  
           - 회고 및 댓글을 어떻게 작성해야 하는지 질문  
           - 회고 및 댓글 관련 제출 마감일 문의  

        10. **메모어 슬랙 사용 가이드**:  
            - 슬랙 사용법, 채널 활용 방법에 대한 질문  

        11. **회고/댓글 제출 가이드**:  
            - 회고 또는 댓글을 제출하는 방법, 일정 관련 문의  

        12. **모임 관련**:  
            - 모임의 기본적인 진행 방식, 운영 방식 질문  

        13. **모임 편성 관련**:  
            - 특정 모임에 편성되었는지 여부 확인  
            - 모임 구성 방식에 대한 질문  

        14. **모임 변경 관련**:  
            - 신청한 모임 날짜나 시간을 변경하고 싶다는 요청  
            - 모임을 취소하고 다른 일정으로 변경하고 싶은 경우  

        15. **오프라인 모임 장소 관련**:  
            - 오프라인 모임이 어디에서 진행되는지 질문  
            - 장소 변경 여부 확인  

        16. **오프라인 및 온라인 비용**:  
            - 오프라인과 온라인 참가 비용 관련 문의  

        17. **취소/환불 관련**:  
            - 보증금 환불 요청, 모임 취소 후 환불 여부 질문  

        18. **메모어라이브 관련**:  
            - 메모어라이브 관련 질문  
            - 메모어라이브 참여 방법, 일정 관련 문의  

        19. **신청 확인**:  
            - 메모어라이브 참가 신청이 정상적으로 접수되었는지 확인 요청  
            - 본인이 신청한 메모어라이브 내역 조회 및 신청 상태 확인  

        20. **참가 조건**:  
            - 메모어라이브 참가 대상 및 필수 자격 요건 문의 

        21. **메모어아카이빙 관련**:  
            - 메모어 아카이빙(기록 저장) 관련 문의  


        대화의 모든 내용을 고려해서, 적절한 범주를 모두 찾아야 해.
        """),
       ("user", "#Format: {format_instructions}\n\n#Question: {question}#"),
    ]
)

parser = JsonOutputParser(pydantic_object=AICategory)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

def get_keyword(query):

    # 모델 초기화
    # temperature이 0에 가까울 수록 정확도가 높은 답변만을 선택함 (1에 가까울수록 랜덤성을 가미함)
    model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo", api_key=api_key)

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

    # 정제 얼마정도 되었는지 확인
    # print(message_string)

    # AI 모델을 통해 키워드 추출
    keyword_result = remove_duplicates(get_keyword(message_string)["category"])

    # print("Before >>>> ", keyword_result)
    filtered_categories = [category for category in keyword_result if category in chatbot_options]
    # print("After >>>> ", filtered_categories)

    # print(">>>>>>>>>>>>> ✅ AI 키워드 추출 성공 <<<<<<<<<<<<<<")
    return filtered_categories


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
        if (message in chatbot_options):
            chatbot_dict[message] += 1


    return chatbot_dict

def text_category(file_path, text_dict):

    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    df = pd.DataFrame(df)


    keywords_list = process_csv(file_path)

    for keywords in keywords_list:
        if keywords in chatbot_options:
            text_dict[keywords] += 1
            

    return text_dict

# print(text_category("memoir_조혜미.csv", chatbot_dict_not_use))


grouping = {
    "메모어 신청 관련": ["메모어 신청 관련", "슬랙 초대링크 관련", "참가 신청 메시지 관련", "슬랙 이메일 변경", "모집 기간 관련"],

    "보증금 관련": ["보증금 관련", "멤버쉽/보증금 제도 관련", "멤버쉽/보증금 확인 관련"],

    "회고 및 댓글 작성 관련": ["회고 및 댓글 작성 관련", "메모어 슬랙 사용 가이드", "회고/댓글 제출 가이드"],

    "모임 관련": ["모임 관련", "모임 편성 관련", "모임 변경 관련", "오프라인 모임 장소 관련", "오프라인 및 온라인 비용"],

    "취소/환불 관련": ["취소/환불 관련"],

    "메모어라이브 관련": ["메모어라이브 관련", "신청 확인", "참가 조건"],

    "메모어아카이빙 관련": ["메모어아카이빙 관련"],


}

def group_category(small_dict, big_dict):
    for s_key, s_value in small_dict.items():
        if s_value > 0:  # 값이 0보다 클 때만
            for group_key, group_values in grouping.items():
                if s_key in group_values:  
                    big_dict[group_key] += s_value  
                    break  
    
    return big_dict  