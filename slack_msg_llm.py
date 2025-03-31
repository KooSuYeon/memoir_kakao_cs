import os
import zipfile
import torch
import pandas as pd
from dotenv import load_dotenv
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import re

openai.api_key = os.getenv("OPEN_API_KEY")
# KoSentenceBERT 모델 로드
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

openai.log = None  

class Keywords(BaseModel):
    description: List[str] = Field(description="AI가 생성한 키워드 리스트")


load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
parser = JsonOutputParser(pydantic_object=Keywords)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        너는 자기소개를 분석하는 AI야. 
        주어진 문장에서 핵심적인 키워드를 뽑아줘. 
        키워드는 단어 또는 짧은 구 형태로 구성되어야 해.
        키워드는 중복되지 않도록 해야 하며, 최대 20개까지만 뽑아야 해.
        키워드의 개수는 전체 문장의 50%로 설정해야 해.
        예를 들어, 10줄의 대화가 주어지면 5개의 키워드를 추출해야 해.
        키워드에 "메모어"가 포함되지 않도록 해줘.
        """),
       ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

def get_keyword(query):

    model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", api_key=api_key)

    chain = prompt | model | parser

    response = chain.invoke({"question": query})

    return response


def process_csv(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)

    users = set(df["User"])  
    users_keywords = []

    for user in users:
        user_messages = df[df["User"] == user]["Message"].tolist()
        
        all_keywords = []
        for message in user_messages:
            keywords = get_keyword(message)
            
            if isinstance(keywords, dict) and "description" in keywords:
                all_keywords.extend(keywords["description"])
            elif isinstance(keywords, str):
                all_keywords.append(list(keywords))
            elif isinstance(keywords, list):  # 리스트인 경우 추가
                all_keywords.extend(keywords)
            else:
                print(f"Warning: Unexpected format for message {type(keywords)}, '{message}': {keywords}")  # 디버깅 출력
                all_keywords.append("No description")
        
        users_keywords.append({"User": user, "Keywords":all_keywords})

    keywords_df = pd.DataFrame(users_keywords)
    keywords_df.to_csv('user_keywords_new_2.csv', index=False, encoding='utf-8')
    print("CSV 파일로 저장 완료: user_keywords_new_2.csv")

# process_csv("slack_messages.csv")


# 커피챗 매칭 알고리즘 고안
def get_user_keywords(userId, file_path):

    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    # userId의 유저의 키워드가 csv 파일의 다른 유저의 키워드와 유사확률을 구해야 함
    
    sender_row = df[df["User"] == userId]
    sender_keywords = sender_row["Keywords"].values[0]
    sender_keywords = ast.literal_eval(sender_keywords) 
    
    return sender_keywords


def get_user_list(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    user_list = df["User"].tolist() 

    # print("BEFORE >>> ", len(user_list))
    # 유저 키워드가 비어있지 않은 경우만
    # 필터링
    # for user in user_list:
    #     print(user, get_user_keywords(user, file_path))
    # user_list = [user for user in user_list if get_user_keywords(user, file_path)]

    # print("AFTER >>> ", len(user_list))
    return user_list

# get_user_list("user_keywords.csv")
# print(get_user_keywords("U089GQZPE22", "user_keywords.csv"))


def remove_excluded_keywords(keywords, excluded_keywords):
    return [kw for kw in keywords if kw not in excluded_keywords]

def get_sentence_embedding(sentence):
    if not sentence:  
        return torch.zeros(768) 

    sentence_embedding = model.encode([sentence], convert_to_tensor=True)

    return sentence_embedding.squeeze()  

def compute_similarity(emb1, emb2):
    if emb1.dim() == 0:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 0:
        emb2 = emb2.unsqueeze(0)

    similarity = util.pytorch_cos_sim(emb1, emb2)
    return similarity.item()

import re

# def find_top_keywords(input_keywords, candidate_keywords, excluded_keywords, top_n=3):
#     top_matched_keywords = []

#     def clean_keywords(keywords):
#         cleaned_keywords = []
#         for kw in keywords:
#             kw = kw.strip()

#             if kw in excluded_keywords:
#                 continue
            
#             if re.search(r'\d+기', kw): 
#                 continue

#             cleaned_keywords.append(kw)
#         return cleaned_keywords

#     input_keywords = clean_keywords(input_keywords)
#     candidate_keywords = clean_keywords(candidate_keywords)

#     input_embeddings = {kw: get_sentence_embedding(kw) for kw in input_keywords}
#     candidate_embeddings = {kw: get_sentence_embedding(kw) for kw in candidate_keywords}

#     similarity_scores = []
#     for input_kw, input_emb in input_embeddings.items():
#         for candidate_kw, candidate_emb in candidate_embeddings.items():
#             sim = compute_similarity(input_emb, candidate_emb)
#             similarity_scores.append((input_kw, candidate_kw, sim))

#     similarity_scores.sort(key=lambda x: x[2], reverse=True)
#     top_matched_keywords = similarity_scores[:top_n]

#     return top_matched_keywords


class S_keywordList(BaseModel):
    keywords: List[str]

s_parser = JsonOutputParser(pydantic_object=S_keywordList)

# 기존 프롬프트 정의
s_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        너는 두 유저의 키워드들을 바탕으로 유사도가 높은 키워드를 3개를 추출해주는 AI야. 
        유의미하고 실용적인 키워드들을 우선적으로 추출해줘. 즉, 단순히 유사도가 높은 키워드보다는 
        두 유저 간에 공통적인 관심사를 나타낼 수 있는, 실제로 유용하게 활용할 수 있는 키워드를 추출해야 해.

        ** 금지 사항 **
        - 키워드들끼리는 중복되지 않게 해줘
        - 키워드에 "메모어"가 포함되지 않도록 해줘.
        - 키워드에 "회고"가 포함되지 않도록 해줘.
        - 키워드에는 사람의 이름이 포함되지 않도록 해줘.

        """),
       ("user", "#Format: {format_instructions}\n\n#User_1: {user_1_keywords}\n\n#User_2: {user_2_keywords}\n\n"),
    ]
)

s_prompt = s_prompt.partial(format_instructions=s_parser.get_format_instructions())

# ChatGPT 응답을 Pydantic 모델로 파싱하는 함수
def get_simularity_keyword(query1, query2):
    model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", api_key=api_key)

    # Chain을 이용하여 모델에 요청
    chain = s_prompt | model | s_parser

    # 응답을 받아서 Pydantic 모델로 파싱
    response = chain.invoke({"user_1_keywords": query1, "user_2_keywords": query2})["keywords"][:3]

    print(response)
    return response


class Topic(BaseModel):
    description:str

t_parser = JsonOutputParser(pydantic_object=Topic)


t_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        당신은 두 유저의 자기소개를 바탕으로, 커피챗 주제와 활동을 추천하는 AI입니다.  
        대화가 단순히 관심사 공유에서 끝나는 것이 아니라, 실질적인 경험을 나누거나, 서로 새로운 관점을 배울 수 있도록 해야 합니다.  
        특히, 처음 만난 사람들이 부담 없이 이야기할 수 있도록, 친근하고 흥미로운 주제를 제시해야 합니다.  

        ** 🚫 금지 사항 **  
        - **반드시 한국어로만** 대답해야 합니다.  
        - **무조건 존댓말을 사용**해야 합니다.  
        - **반드시 구어체로만** 대답해야 합니다.  
        - **이름은 포함하지 않아야 합니다.**  
        - 추천하는 주제는 서로 겹치지 않도록 합니다.  
        - 지나치게 기술적이거나 어려운 내용은 피하고, 누구나 공감할 수 있는 주제를 추천합니다.  
        """),
        ("user", "#Format: {format_instructions}\n\n#User_1: {user_1_keywords}\n\n#User_2: {user_2_keywords}\n\n"),
    ]
)



t_prompt = t_prompt.partial(format_instructions=t_parser.get_format_instructions())

# ChatGPT 응답을 Pydantic 모델로 파싱하는 함수
def get_recommand_topic(query1, query2):
    model = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo", api_key=api_key)

    # Chain을 이용하여 모델에 요청
    chain = t_prompt | model | t_parser

    # 응답을 받아서 Pydantic 모델로 파싱
    response = chain.invoke({"user_1_keywords": query1, "user_2_keywords": query2})

    print(response)
    return response


def save_embeddings(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)
    df["Keywords"] = df["Keywords"].apply(ast.literal_eval)

    




def get_matchers(userId, keywords_list, file_path, similarity_threshold=0.8):
    df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="replace", header=0)

    # Keywords 컬럼을 리스트로 변환
    df["Keywords"] = df["Keywords"].apply(ast.literal_eval)

    excluded_keywords = ["메모어", "회고"]

    keywords_list_filtered = remove_excluded_keywords(keywords_list, excluded_keywords)

    # 입력 키워드 임베딩 계산
    input_text = " ".join(keywords_list_filtered)
    input_embedding = get_sentence_embedding(input_text)

    similarities = []
    for keywords in df["Keywords"]:
        keyword_text = " ".join(keywords)  
        keyword_embedding = get_sentence_embedding(keyword_text)
        similarity = compute_similarity(input_embedding, keyword_embedding)
        similarities.append(similarity)

    # 유사도 결과 추가
    df["Similarity"] = similarities

    
    # 본인 제외 & 임계값 이상 필터링
    matched_users = df[(df["Similarity"] >= similarity_threshold) & (df["User"] != userId)][["User", "Keywords", "Similarity"]][:6]

    """
    각각
    """
    # matched_results = []
    # for index, row in matched_users.iterrows():
    #     user = row["User"]
    #     candidate_keywords = row["Keywords"]
        
    #     # 각 후보자에 대해 Top Keywords를 추출
    #     top_keywords = get_simularity_keyword(keywords_list, candidate_keywords)
        
    #     # 결과에 Top Keywords 추가
    #     matched_results.append({
    #         "User": user,
    #         "Keywords": candidate_keywords,
    #         "Similarity": row["Similarity"],
    #         "Top Keywords": top_keywords
    #     })

    """
    한번에
    """
    all_keywords = [keyword for keywords in matched_users["Keywords"] for keyword in keywords]
    top_keywords = get_simularity_keyword(keywords_list, all_keywords)


    slack_msg_df = pd.read_csv("slack_messages.csv", encoding="utf-8", encoding_errors="replace", header=0)
    user_messages = slack_msg_df[slack_msg_df["User"] == userId]["Message"]

    # 결과에 Top Keywords 추가 (모든 후보자에게 동일한 Top Keywords)
    matched_results = []
    for index, row in matched_users.iterrows():
        user = row["User"]
        candidate_messages = slack_msg_df[slack_msg_df["User"] == user]["Message"]
        candidate_keywords = row["Keywords"]
        # recommand_topic = get_recommand_topic(user_messages, candidate_messages)["description"]
        
        # 결과에 Top Keywords 추가
        matched_results.append({
            "User": user,
            "Keywords": candidate_keywords,
            "Similarity": row["Similarity"],
            "Top Keywords": top_keywords,
            # "AI Recommand" : recommand_topic
        })

    # 유사도 높은 순으로 정렬 후 Top 6 추출
    matched_results = sorted(matched_results, key=lambda x: x["Similarity"], reverse=True)

    return matched_results

    # matched_results = []

    # for index, row in df.iterrows():
    #     user = row["User"]
    #     candidate_keywords = row["Keywords"]

    #     candidate_text = " ".join(candidate_keywords)
    #     candidate_embedding = get_sentence_embedding(candidate_text)

    #     similarity = compute_similarity(input_embedding, candidate_embedding)

    #     if similarity >= similarity_threshold and user != userId:
    #         top_keywords = get_simularity_keyword(keywords_list, candidate_keywords)
    #         matched_results.append({"User": user, "Keywords": candidate_keywords, "Similarity": similarity, "Top Keywords": top_keywords})

    # # 유사도 높은 순으로 정렬 후 Top 6 추출
    # matched_results = sorted(matched_results, key=lambda x: x["Similarity"], reverse=True)[:6]

    # return matched_results


# userId = "U089GQZPE22"
# matched_users = get_matchers(userId, get_user_keywords(userId, "user_keywords_new.csv"), "user_keywords_new.csv", 0.6)
# print(matched_users)



"""
한국어 유사도 측정 디버깅 ZONE
"""
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # 문서 정의
# documents = [
#     "과학과 예술에 관심",
#     "과학에 관심"
# ]

# # TfidfVectorizer로 텍스트 벡터화
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(documents)

# # 코사인 유사도 계산
# cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
# print("Cosine Similarity:", cosine_sim[0][0])


"""
KOBERT
"""
# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # KoBERT 모델과 토크나이저 로드
# model_name = 'monologg/kobert'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

# # 문장을 벡터로 변환하는 함수
# def get_sentence_embedding(sentence):
#     # 문장 토큰화
#     inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
#     # 모델을 통해 임베딩을 얻음
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # [CLS] 토큰의 임베딩을 사용
#     sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
#     return sentence_embedding

# # 두 문장의 유사도를 측정하는 함수
# def compute_similarity(sentence1, sentence2):
#     emb1 = get_sentence_embedding(sentence1)
#     emb2 = get_sentence_embedding(sentence2)
    
#     # 코사인 유사도 계산
#     similarity = cosine_similarity([emb1], [emb2])
    
#     return similarity[0][0]

# # 테스트 예시
# sentence1 = "과학과 예술에 관심"
# sentence2 = "사랑하는 사람과 여행 가고 싶다"
# similarity = compute_similarity(sentence1, sentence2)

# print(f"유사도: {similarity}")

"""
KoSentenceBERT 모델
"""


# from sentence_transformers import SentenceTransformer, util

# # KoSentenceBERT 모델 로드
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# # 문장 임베딩 함수
# def get_sentence_embedding(sentence):
#     return model.encode(sentence, convert_to_tensor=True)

# # 두 문장의 유사도를 측정하는 함수
# def compute_similarity(sentence1, sentence2):
#     emb1 = get_sentence_embedding(sentence1)
#     emb2 = get_sentence_embedding(sentence2)

#     # 코사인 유사도 계산
#     similarity = util.pytorch_cos_sim(emb1, emb2)

#     return similarity.item()

# # 테스트 예시
# sentence1 = "과학과 예술에 관심"
# sentence2 = "과학에 관심"
# similarity = compute_similarity(sentence1, sentence2)

# print(f"유사도: {similarity}")

