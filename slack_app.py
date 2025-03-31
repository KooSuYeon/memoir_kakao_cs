import streamlit as st
from slack_msg_llm import get_user_list, get_matchers, get_user_keywords
import pandas as pd

st.set_page_config(page_title="자기소개 키워드 기반 커피챗 추천 서비스", layout="wide")

st.title("☕️ 자기소개 키워드 기반 커피챗 추천 서비스입니다!")

# 📌 사용자 리스트 불러오기
keywords_file = "user_keywords_new.csv"
user_list = get_user_list(keywords_file)

# 📌 드롭다운 선택 박스 추가
selected_user = st.selectbox("유저를 선택하세요", user_list, index=0)

# 📌 선택된 유저 표시
st.write(f"✅ 선택된 유저: **{selected_user}**")

selected_user_keywords = get_user_keywords(selected_user, keywords_file)
st.markdown(" ".join([f"`{keyword}`" for keyword in selected_user_keywords]))

st.markdown("---")

st.text("🔎 선택된 유저의 커피챗 후보를 추천합니다!")

matched_users = get_matchers(selected_user, get_user_keywords(selected_user, keywords_file), keywords_file, 0.6)

# 리스트를 DataFrame으로 변환
df_matched_users = pd.DataFrame(matched_users)

# Similarity 열까지만 가져오기
df_matched_users_filtered = df_matched_users[['User', 'Keywords', 'Similarity']]

# Top Keywords를 따로 추출하여 마지막에 추가
top_keywords_list = matched_users[0]["Top Keywords"]

# Streamlit에서 DataFrame 출력
st.dataframe(df_matched_users_filtered)
st.write("🔑 Top Keywords:")
st.table(top_keywords_list)
