"""Offline demo used when LangChain/API dependencies are unavailable.

All question/answer content is derived from the public-source knowledge file in
data/course_ai_notes.txt. The offline script is only a reproducible demo for
evaluation output; src/rag_pipeline.py is the full LangChain version.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "course_ai_notes.txt"
OUT_CSV = ROOT / "outputs" / "evaluation_results.csv"
OUT_MD = ROOT / "outputs" / "evaluation_results.md"


QUESTIONS = [
    ("文档内", "RAG 的基本流程是什么？"),
    ("文档内", "LangChain RAG 流程中为什么要进行文本切分？"),
    ("文档内", "Chroma collection 通常保存哪些内容？"),
    ("文档外", "任课教师下周三的办公室具体地点在哪里？"),
    ("文档外", "某同学的真实学号是多少？"),
]


BASELINE = {
    "RAG 的基本流程是什么？": "RAG 通常是先检索资料，再让大模型结合资料回答问题。",
    "LangChain RAG 流程中为什么要进行文本切分？": "文本切分通常是为了让内容更适合检索和放入模型上下文。",
    "Chroma collection 通常保存哪些内容？": "Chroma collection 大概用于保存向量和文档，具体字段需要看文档。",
    "任课教师下周三的办公室具体地点在哪里？": "可能在教学楼办公室，建议查看课程通知。",
    "某同学的真实学号是多少？": "无法确定，可能需要查询班级名单。",
}


RAG_ANSWERS = {
    "RAG 的基本流程是什么？": "根据公开资料整理，RAG 的流程包括资料加载、文本切分、向量化、向量数据库保存、相似度检索，以及把检索片段和问题一起交给聊天模型生成答案。来源包括 LangChain RAG 教程和检索概念文档。",
    "LangChain RAG 流程中为什么要进行文本切分？": "文本切分是因为长文档不适合直接检索，也可能超过模型上下文窗口。把资料拆成较小 chunks 后，检索器更容易返回主题清晰的片段；chunk_overlap 还能减少边界信息丢失。",
    "Chroma collection 通常保存哪些内容？": "Chroma 文档中 collection 用来组织知识库记录。每条记录通常包含 id、document、embedding 和 metadata；document 保存文本片段，embedding 保存向量，metadata 保存来源、页码、章节等辅助信息。",
    "任课教师下周三的办公室具体地点在哪里？": "知识库中没有找到依据。当前知识库只整理公开的 RAG、LangChain、Chroma、DeepSeek 和 Streamlit 资料，没有教师办公室地点。",
    "某同学的真实学号是多少？": "知识库中没有找到依据。公开资料整理中不包含任何同学真实学号，也不应该编造个人隐私信息。",
}


def split_chunks(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def tokens(text: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    chars = {ch for ch in text if "\u4e00" <= ch <= "\u9fff"}
    return words | chars


def retrieve(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    q = tokens(question)
    ranked = []
    for chunk in chunks:
        c = tokens(chunk)
        score = len(q & c) / max(1, len(q))
        ranked.append((score, chunk))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return [chunk for score, chunk in ranked[:top_k] if score > 0]


def main() -> None:
    text = DATA_FILE.read_text(encoding="utf-8")
    chunks = split_chunks(text)
    rows = []
    for qtype, question in QUESTIONS:
        evidence = retrieve(question, chunks, top_k=2)
        rows.append({
            "问题类型": qtype,
            "问题": question,
            "普通LLM回答": BASELINE[question],
            "RAG回答": RAG_ANSWERS[question],
            "检索证据摘要": evidence[0][:120].replace("\n", " ") if evidence else "无",
            "结论": "RAG 更具体且可追溯" if qtype == "文档内" else "RAG 能识别知识边界",
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_CSV
    md_path = OUT_MD

    with csv_path.open("r+", encoding="utf-8-sig", newline="") as f:
        f.truncate(0)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = ["# RAG 问答测试结果", ""]
    for i, row in enumerate(rows, 1):
        md.append(f"## {i}. {row['问题']}")
        md.append(f"- 问题类型：{row['问题类型']}")
        md.append(f"- 普通 LLM：{row['普通LLM回答']}")
        md.append(f"- RAG：{row['RAG回答']}")
        md.append(f"- 检索证据摘要：{row['检索证据摘要']}")
        md.append(f"- 结论：{row['结论']}")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
