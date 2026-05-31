"""LangChain RAG pipeline for the AI general course knowledge base."""

from __future__ import annotations

import os
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "course_ai_notes.txt"
DB_DIR = ROOT / "chroma_db"


PROMPT = PromptTemplate.from_template(
    """你是课程知识库问答助手。请只依据【参考资料】回答【问题】。
如果参考资料没有答案，请明确说“知识库中没有找到依据”，不要编造。

【参考资料】
{context}

【问题】
{question}

【回答】"""
)


def build_retriever(chunk_size: int = 500, chunk_overlap: int = 80, top_k: int = 3):
    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    return vectordb.as_retriever(search_kwargs={"k": top_k})


def build_chain():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY 环境变量。")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=build_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )


if __name__ == "__main__":
    qa = build_chain()
    questions = [
        "RAG 的基本流程是什么？",
        "本课程实践项目评分标准是什么？",
        "chunk_size 设为 200、500、1000 时有什么差异？",
        "任课教师下周三的办公室具体地点在哪里？",
        "某同学的真实学号是多少？",
    ]
    for question in questions:
        result = qa.invoke({"query": question})
        print("\nQ:", question)
        print("A:", result["result"])
        print("Sources:", [doc.metadata for doc in result["source_documents"]])
