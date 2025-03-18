import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from kakao_msg_llm import process_csv, chatbot_category

st.set_page_config(page_title="카카오톡 CS 분석", layout="wide")
font_path = os.path.join(os.getcwd(), "assets", "fonts", "NotoSansKR-VariableFont_wght.ttf")

# 📊 카카오톡 채널 C/S 채널 csv 파일들의 압축 폴더 업로드
st.title("📦 카카오톡 채널 C/S 채널 csv 파일들의 압축 폴더를 업로드 해주세요")
# 진행도 표시할 때 사용하는 변수
progress = st.progress(0)
status_text = st.empty()

def typing_animation(placeholder):
    # 점점 증가하는 타이핑 애니메이션 효과
    dots = ["🔎", "🧑‍💻", "🧐", "✨"]
    for dot in dots:
        placeholder.text(f"로딩 중{dot}")  # 텍스트 업데이트
        time.sleep(0.5)  # 0.5초마다 텍스트를 변경

# 🔹 압축 파일 업로드
uploaded_file = st.file_uploader("압축 파일을 선택하세요", type="zip")

def get_csv_files_from_folder(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):  # 모든 하위 디렉토리 포함
        # __MACOSX 폴더 제외
        if '__MACOSX' in dirs:
            dirs.remove('__MACOSX')  # 탐색에서 제외
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))  # 전체 경로로 저장
    return csv_files

if uploaded_file is not None:
    chatbot_dict = {"메모어 신청 관련": 0, "슬랙 초대링크 관련": 0, "참가 신청 메시지 관련": 0, "슬랙 이메일 변경": 0, "모집 기간 관련": 0, "보증금 관련": 0,"멤버쉽/보증금 제도 관련": 0, "멤버쉽/보증금 확인 관련": 0, "회고 및 댓글 관련": 0, "메모어 슬랙 사용 가이드": 0, "회고/댓글 제출 가이드": 0,  "모임 관련": 0, "모임 편성 관련": 0, "모임 변경 관련":0, "오프라인 모임 장소 관련": 0, "오프라인 및 온라인 비용": 0, "취소/환불 관련": 0, "메모어라이브 관련" : 0, "신청 확인":0, "참가 조건":0, "메모어아카이빙 관련": 0}
    temp_dir = "temp_extracted_folder"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # ZIP 파일을 임시 디렉토리에 압축 해제
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 압축 해제한 폴더 내 CSV 파일 목록 가져오기
        csv_files = get_csv_files_from_folder(temp_dir)
        total_files = len(csv_files)
        progress_value = 0

        for csv_file in csv_files:
            chatbot_dict = chatbot_category(csv_file, chatbot_dict)

        st.markdown("---")
        st.header("챗봇/상담 키워드 별 챗봇 CS 정보")

        # 딕셔너리를 데이터프레임으로 변환
        df = pd.DataFrame(list(chatbot_dict.items()), columns=["Category", "Count"])

        # Count의 총합을 계산
        total_count = df["Count"].sum()

        # 총합을 마지막 행에 추가
        total_row = pd.DataFrame([["Total", total_count]], columns=["Category", "Count"])
        df_with_total = pd.concat([df, total_row], ignore_index=True)

        # 총합을 제외한 부분만 내림차순으로 정렬
        df_sorted = df_with_total.iloc[:-1].sort_values(by="Count", ascending=False)

        # 총합 행을 맨 마지막에 추가
        df_sorted = pd.concat([df_sorted, total_row], ignore_index=True)

        # # 테이블로 출력
        # st.table(df_sorted)
        
        # st.markdown("---")

        # 키워드를 모을 리스트
        all_keywords = []

        # 컬럼을 두 개로 나누어 배치 (왼쪽, 오른쪽)
        col1, col2 = st.columns([2, 3])  # 왼쪽(작은) 2, 오른쪽(큰) 3 비율로 배치

        # 왼쪽 컬럼에 챗봇 CS 정보를 출력
        with col1:
            st.write("### 챗봇 CS 정보")
            # 딕셔너리를 데이터프레임으로 변환하여 출력
            st.table(df_sorted)

        # 오른쪽 컬럼에 워드 클라우드를 출력
        with col2:
            st.write("### 상담 키워드 워드 클라우드:")
            
            # 워드 클라우드 로딩 중 표시용 빈 영역
            loading_placeholder = st.empty()
            typing_animation(loading_placeholder)  # 타이핑 애니메이션 적용

        # 모든 CSV 파일에 대해 키워드 추출
        for i, csv_file in enumerate(csv_files):
            result = process_csv(csv_file)

            if isinstance(result, dict):  
                keywords = result.get("description", [])  # 키가 없으면 빈 리스트 반환
            elif isinstance(result, list):  
                keywords = result[0].get("description", []) if result and isinstance(result[0], dict) else []
            else:
                keywords = []  # 기본값 설정
            all_keywords.extend(keywords)  # 추출된 키워드를 all_keywords 리스트에 추가
            progress_value = int((i + 1) / total_files * 100)  # 진행도 계산
            progress.progress(progress_value)  # 진행도 업데이트
            status_text.text(f"Extracting keywords from {i + 1}/{total_files} files...")  # 상태 업데이트


        # 워드클라우드 생성
        wordcloud = WordCloud(
            font_path=font_path,  # Mac에서 AppleGothic 폰트 경로 예시
            width=1000,
            height=600,
            background_color='white'
        ).generate(' '.join(all_keywords))

        # 워드 클라우드 생성 후, placeholder에 워드 클라우드를 표시
        with col2:
            loading_placeholder.empty()  # 기존 텍스트 지우기
            plt.figure(figsize=(15, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")  # 축 제거
            st.pyplot(plt)  # Streamlit에 워드클라우드 출력
        # # categorized_keywords는 카테고리별로 키워드가 정리된 딕셔너리
        # categorized_keywords = categorize_keywords_with_spacy(all_keywords)

        # # 기타 카테고리를 제외하고, 나머지 카테고리들에 대해 키워드 개수로 랭킹 정렬
        # sorted_categories = {category: keywords for category, keywords in sorted(categorized_keywords.items(), key=lambda item: len(item[1]), reverse=True) if category != "기타"}

        # # 제목 추가
        # st.title("🔑 카테고리별 키워드 목록 (랭킹 순)")

        # # 카테고리별로 키워드 출력
        # for category, keywords in sorted_categories.items():
        #     # 각 카테고리 제목을 섹션으로 구분하여 강조
        #     st.subheader(f"### {category} 키워드")
            
        #     # 카테고리별 키워드 리스트
        #     if keywords:
        #         st.write("• " + "\n• ".join(keywords))  # 리스트 형식으로 출력
        #     else:
        #         st.write("키워드가 없습니다.")  # 키워드가 없을 경우

    finally:
        # 작업이 끝난 후 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)
