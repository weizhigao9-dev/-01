"""Simple Streamlit interface for the RAG project."""

from pathlib import Path

import streamlit as st

from offline_demo import RAG_ANSWERS, retrieve, split_chunks


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "course_ai_notes.txt"

st.set_page_config(page_title="RAG 知识库问答助手", layout="wide")
st.title("RAG 知识库问答助手")

text = DATA_FILE.read_text(encoding="utf-8")
chunk_size = st.sidebar.slider("chunk_size", 200, 1000, 500, 100)
top_k = st.sidebar.slider("top_k", 1, 5, 3, 1)
show_sources = st.sidebar.checkbox("显示来源片段", value=True)

question = st.text_input("请输入问题", "RAG 的基本流程是什么？")
if st.button("提问"):
    answer = RAG_ANSWERS.get(question, "知识库中没有找到依据，当前离线演示只内置了测试集问题。")
    st.subheader("回答")
    st.write(answer)
    if show_sources:
        st.subheader("检索片段")
        for i, chunk in enumerate(retrieve(question, split_chunks(text, chunk_size), top_k=top_k), 1):
            st.markdown(f"**片段 {i}**")
            st.write(chunk[:500])
