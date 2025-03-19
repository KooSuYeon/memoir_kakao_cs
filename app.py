import streamlit as st
import pandas as pd
import zipfile
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

import time
from kakao_msg_llm import process_csv, chatbot_category, text_category, group_category

st.set_page_config(page_title="카카오톡 CS 분석", layout="wide")
font_path = os.path.join(os.getcwd(), "assets", "fonts", "NotoSansKR-VariableFont_wght.ttf")
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Noto Sans KR']  # Noto Sans KR 폰트 설정
font_manager.fontManager.addfont(font_path)

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

    c_group_dict = {"메모어 신청 관련": 0, "보증금 관련": 0, "회고 및 댓글 작성 관련": 0, "모임 관련": 0, "취소/환불 관련": 0, "메모어라이브 관련": 0, "메모어아카이빙 관련": 0}

    t_group_dict = {"메모어 신청 관련": 0, "보증금 관련": 0, "회고 및 댓글 작성 관련": 0, "모임 관련": 0, "취소/환불 관련": 0, "메모어라이브 관련": 0, "메모어아카이빙 관련": 0}

    chatbot_dict = {"메모어 신청 관련": 0, "슬랙 초대링크 관련": 0, "참가 신청 메시지 관련": 0, "슬랙 이메일 변경": 0, "모집 기간 관련": 0, "보증금 관련": 0,"멤버쉽/보증금 제도 관련": 0, "멤버쉽/보증금 확인 관련": 0, "회고 및 댓글 작성 관련": 0, "메모어 슬랙 사용 가이드": 0, "회고/댓글 제출 가이드": 0,  "모임 관련": 0, "모임 편성 관련": 0, "모임 변경 관련":0, "오프라인 모임 장소 관련": 0, "오프라인 및 온라인 비용": 0, "취소/환불 관련": 0, "메모어라이브 관련" : 0, "신청 확인":0, "참가 조건":0, "메모어아카이빙 관련": 0}

    text_dict = {"메모어 신청 관련": 0, "슬랙 초대링크 관련": 0, "참가 신청 메시지 관련": 0, "슬랙 이메일 변경": 0, "모집 기간 관련": 0, "보증금 관련": 0,"멤버쉽/보증금 제도 관련": 0, "멤버쉽/보증금 확인 관련": 0, "회고 및 댓글 작성 관련": 0, "메모어 슬랙 사용 가이드": 0, "회고/댓글 제출 가이드": 0,  "모임 관련": 0, "모임 편성 관련": 0, "모임 변경 관련":0, "오프라인 모임 장소 관련": 0, "오프라인 및 온라인 비용": 0, "취소/환불 관련": 0, "메모어라이브 관련" : 0, "신청 확인":0, "참가 조건":0, "메모어아카이빙 관련": 0}

    chatbot_large_options = ["메모어 신청 관련", "보증금 관련", "회고 및 댓글 작성 관련", "모임 관련", "취소/환불 관련", "메모어라이브 관련", "메모어아카이빙 관련"]

    grouping = {
        "메모어 신청 관련": ["메모어 신청 관련", "슬랙 초대링크 관련", "참가 신청 메시지 관련", "슬랙 이메일 변경", "모집 기간 관련"],
        "보증금 관련": ["보증금 관련", "멤버쉽/보증금 제도 관련", "멤버쉽/보증금 확인 관련"],
        "회고 및 댓글 작성 관련": ["회고 및 댓글 작성 관련", "메모어 슬랙 사용 가이드", "회고/댓글 제출 가이드"],
        "모임 관련": ["모임 관련", "모임 편성 관련", "모임 변경 관련", "오프라인 모임 장소 관련", "오프라인 및 온라인 비용"],
        "취소/환불 관련": ["취소/환불 관련"],
        "메모어라이브 관련": ["메모어라이브 관련", "신청 확인", "참가 조건"],
        "메모어아카이빙 관련": ["메모어아카이빙 관련"],
    }

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
        
        c_group_dict = group_category(chatbot_dict, c_group_dict)

        print(c_group_dict)
        st.markdown("---")
        st.text("📢 챗봇/살담 키워드 분포는 키워드의 카테고리별 분포를 시각화한 것입니다.\n챗봇의 Count의 기준은 챗봇 내 버튼 각각의 Count이며\n상담의 Count의 기준은 상담 내용에서 챗봇 내 버튼을 기준으로 분류한 것의 Count입니다.")

        st.text("📢 상담 키워드 제외되는 메시지 : 챗봇 정보때 사용되었던 버튼 텍스트 정보, 메모어 주간 정기 메시지 (”[메모어 XX기 X주차]…”), 대표님 메시지를 제외한 봇 메시지")

        st.header("챗봇/상담 CS 키워드 분포")

        # 딕셔너리를 데이터프레임으로 변환
        chatbot_df = pd.DataFrame(list(chatbot_dict.items()), columns=["Category", "Count"])
        chatbot_df = chatbot_df[~chatbot_df["Category"].isin(chatbot_large_options)]

        # Count의 총합을 계산
        c_total_count = chatbot_df["Count"].sum()

        # 총합을 마지막 행에 추가
        c_total_row = pd.DataFrame([["Total", c_total_count]], columns=["Category", "Count"])
        c_df_with_total = pd.concat([chatbot_df, c_total_row], ignore_index=True)

        # 총합을 제외한 부분만 내림차순으로 정렬
        c_df_sorted = c_df_with_total.iloc[:-1].sort_values(by="Count", ascending=False)
        # 총합 행을 맨 마지막에 추가
        c_df_sorted = pd.concat([c_df_sorted, c_total_row], ignore_index=True)

        def assign_group(category):
            # 각 카테고리가 어떤 그룹에 속하는지 반환
            for group_name, categories in grouping.items():
                if category in categories:
                    return group_name
            return ""  # 그룹에 속하지 않는 경우 "기타"로 설정
        
        c_df_sorted["Group"] = c_df_sorted["Category"].apply(assign_group)

        # 키워드를 모을 리스트
        all_keywords = []

        # 진행 상태 초기화
        progress.progress(0)
        typing_animation(status_text)  # 로딩 애니메이션 추가

        for i, csv_file in enumerate(csv_files):
            text_dict = text_category(csv_file, text_dict)
            progress_value = int(((i + 1) / total_files) * 100)
            progress.progress(progress_value)  # 진행 상태 업데이트

        t_group_dict = group_category(text_dict, t_group_dict)

        # 두 딕셔너리의 그룹 순서를 동일하게 맞추기 위해 그룹 이름을 추출
        categories = list(c_group_dict.keys())

        # 수평 막대그래프를 그리기 위한 데이터 준비
        c_values = list(c_group_dict.values())
        t_values = list(t_group_dict.values())

        # 그래프 크기 설정
        fig, ax = plt.subplots(figsize=(10, 6))

        # 양방향 막대 그래프 그리기
        bars_c = ax.barh(categories, c_values, color='skyblue', label='Chatbot')
        bars_t = ax.barh(categories, [-t for t in t_values], color='salmon', label='Text')


        # y축과 x축 레이블 설정
        ax.set_xlabel('Count')
        ax.set_title('챗봇과 텍스트 CS 분포')

        # 범례 추가
        ax.legend()

        for bar in bars_c:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{width:.0f}', va='center', ha='left', fontsize=10, color='black')

        for bar in bars_t:
            width = bar.get_width()
            ax.text(width - 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{-width:.0f}', va='center', ha='right', fontsize=10, color='black')


        # 그래프 표시
        st.pyplot(fig)

        st.markdown("---")
        st.text("📢 아래는 세부 CS정보입니다. 이는 큰 범주 (메모어 신청 관련, 보증금 관련, 모임 관련, 취소/환불 관련, 회고 및 댓글 작성 관련, 메모어라이브 관련, 메모어아카이빙 관련)을 제외하고 하위 사항만을 표시했으며 큰 범주는 Group 섹션에 상위 개념을 표시해두었습니다.")
        # 컬럼을 두 개로 나누어 배치 (왼쪽, 오른쪽)
        col1, col2 = st.columns([2, 3])  # 왼쪽(작은) 2, 오른쪽(큰) 3 비율로 배치

        # 왼쪽 컬럼에 챗봇 CS 정보를 출력
        with col1:
            st.write("### 챗봇 CS 정보")
            # 딕셔너리를 데이터프레임으로 변환하여 출력
            st.table(c_df_sorted)

        with col2:
            st.write("### 텍스트 상담 CS 정보:")

            # 딕셔너리를 데이터프레임으로 변환
            text_df = pd.DataFrame(list(text_dict.items()), columns=["Category", "Count"])

            # Count의 총합을 계산
            t_total_count = text_df["Count"].sum()

            # 총합을 마지막 행에 추가
            t_total_row = pd.DataFrame([["Total", t_total_count]], columns=["Category", "Count"])
            t_df_with_total = pd.concat([text_df, t_total_row], ignore_index=True)

            # 총합을 제외한 부분만 내림차순으로 정렬
            t_df_sorted = t_df_with_total.iloc[:-1].sort_values(by="Count", ascending=False)
            t_df_sorted = t_df_sorted[~t_df_sorted['Category'].isin(chatbot_large_options)]

            # 총합 행을 맨 마지막에 추가
            t_df_sorted = pd.concat([t_df_sorted, t_total_row], ignore_index=True)

            t_df_sorted["Group"] = t_df_sorted["Category"].apply(assign_group)

            # 워드 클라우드 로딩 중 표시용 빈 영역
            st.table(t_df_sorted)
    finally:
        # 작업이 끝난 후 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)
