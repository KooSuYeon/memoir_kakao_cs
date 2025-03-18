import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
from kakao_msg_llm import process_csv, categorize_keywords_with_spacy
from wordcloud import WordCloud

# 📊 카카오톡 채널 C/S 채널 csv 파일들의 압축 폴더 업로드
st.title("📦 카카오톡 채널 C/S 채널 csv 파일들의 압축 폴더를 업로드 해주세요")

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
    temp_dir = "temp_extracted_folder"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # ZIP 파일을 임시 디렉토리에 압축 해제
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 압축 해제한 폴더 내 CSV 파일 목록 가져오기
        csv_files = get_csv_files_from_folder(temp_dir)

        # 키워드를 모을 리스트
        all_keywords = []

        # 모든 CSV 파일에 대해 키워드 추출
        for csv_file in csv_files:
            keywords = process_csv(csv_file)["description"]  # 각 CSV 파일에서 키워드 추출
            all_keywords.extend(keywords)  # 추출된 키워드를 all_keywords 리스트에 추가

        # 결과 출력 (키워드 목록)
        st.write("### 추출된 키워드 목록:")
        st.write(all_keywords)

        # 한글 폰트를 적용한 워드클라우드 생성
        wordcloud = WordCloud(
            font_path='/Library/Fonts/AppleGothic.ttf',  # Mac에서 AppleGothic 폰트 경로 예시
            width=800, 
            height=400, 
            background_color='white'
        ).generate(' '.join(all_keywords))

        # 워드 클라우드를 화면에 출력
        st.write("### 키워드 워드 클라우드:")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")  # 축 제거
        st.pyplot(plt)  # Streamlit에 워드클라우드 출력

        # categorized_keywords는 카테고리별로 키워드가 정리된 딕셔너리
        categorized_keywords = categorize_keywords_with_spacy(all_keywords)

        # 기타 카테고리를 제외하고, 나머지 카테고리들에 대해 키워드 개수로 랭킹 정렬
        sorted_categories = {category: keywords for category, keywords in sorted(categorized_keywords.items(), key=lambda item: len(item[1]), reverse=True) if category != "기타"}

        # 제목 추가
        st.title("🔑 카테고리별 키워드 목록 (랭킹 순)")

        # 카테고리별로 키워드 출력
        for category, keywords in sorted_categories.items():
            # 각 카테고리 제목을 섹션으로 구분하여 강조
            st.subheader(f"### {category} 키워드")
            
            # 카테고리별 키워드 리스트
            if keywords:
                st.write("• " + "\n• ".join(keywords))  # 리스트 형식으로 출력
            else:
                st.write("키워드가 없습니다.")  # 키워드가 없을 경우

    finally:
        # 작업이 끝난 후 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)
