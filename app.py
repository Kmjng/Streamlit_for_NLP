# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:12:56 2024

@author: user

# prompt 로 실행 
cd C:/Users/user/Downloads/code/
streamlit run app.py
"""

import streamlit as st
#from fts import * 
from collections import Counter
import re
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords


# 영문 불용어 처리

# NLTK 불용어 다운로드
nltk.download('stopwords')

# 사용자 함수 정의 
def extract_words(text):
    # 정규 표현식을 사용하여 단어만 추출하고 소문자로 변환
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(stopwords.words('english'))
    # 불용어를 제외한 단어 리스트 반환
    return [word for word in words if word not in stop_words]


st.title("Word Count - 영문텍스트 version")
st.subheader("자주 등장한 단어를 찾아 줍니다")

# 사용 방법 플로우 차트
st.subheader("💡사용 방법")
flowchart = """
digraph {
rankdir=LR;  // 가로 방향 설정
node [shape=box, style=filled, fillcolor="#E0E0E0", height=1.5, width=2.5]; // 노드 스타일 및 크기 설정
A [label="사용자 입력" shape=ellipse fillcolor="#A3C1AD" height=1.5 width=2.5 fontsize=18]; // 글씨 크기 설정
B [label="[분석하기] 버튼 클릭" shape=ellipse fillcolor="#A3C1AD" height=1.5 width=2.5 fontsize=18];
C [label="결과 확인" shape=ellipse fillcolor="#A3C1AD" height=1.5 width=2.5 fontsize=18];

A -> B -> C;
}
"""
st.graphviz_chart(flowchart)


# 기능 추가 
st.write("""
1. 사용자 입력: 글을 입력합니다.
2. '분석하기'를 누르면 등장한 단어가 카운트 됩니다. 
3. 영문의 경우, 불용어 처리 기능이 들어갑니다. 
""")

# 사용자 입력 받기
user_input = st.text_area("🟢 문장을 입력하세요:")

# 분석하기 버튼
if st.button("📥분석하기"):
    # 단어 추출
    words = extract_words(user_input)

    # 단어 빈도 계산
    word_counts = Counter(words)

    # 자주 등장한 단어 순서대로 정렬
    most_common_words = word_counts.most_common()

    # 결과 출력
    if most_common_words:
        st.subheader("🗨️자주 등장한 단어:")
        for word, count in most_common_words:
            st.write(f"{word}: {count}회")
       
        
        # 바 그래프 출력
        words, counts = zip(*most_common_words)  # 단어와 카운트를 각각 리스트로 변환
        # Matplotlib 사용
        st.markdown("#### 📊등장 단어 시각화")
        plt.figure(figsize=(10, 5))
        plt.barh(words, counts, color='skyblue')  # 가로 바 그래프
        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.title("Frequency of words")
        plt.xticks(rotation=45)
        plt.gca().invert_yaxis()  # y축 반전
        st.pyplot(plt)  # Streamlit에 Matplotlib 그래프 표시
        
           
    else:
        st.write("입력된 문장에서 단어를 찾을 수 없습니다.")

