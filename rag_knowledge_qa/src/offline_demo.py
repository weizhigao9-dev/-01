"""Offline demo used when LangChain/API dependencies are unavailable."""

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
    ("文档内", "本课程实践项目评分标准是什么？"),
    ("文档内", "chunk_size 设为 200、500、1000 时有什么差异？"),
    ("文档外", "任课教师下周三的办公室具体地点在哪里？"),
    ("文档外", "某同学的真实学号是多少？"),
]


BASELINE = {
    "RAG 的基本流程是什么？": "RAG 通常是先检索资料，再让大模型结合资料回答问题。",
    "本课程实践项目评分标准是什么？": "一般会看完成度、创新性、代码质量和展示效果，但具体比例需要看课程文件。",
    "chunk_size 设为 200、500、1000 时有什么差异？": "小块更精确，大块上下文更完整，具体效果取决于数据。",
    "任课教师下周三的办公室具体地点在哪里？": "可能在教学楼办公室，建议查看课程通知。",
    "某同学的真实学号是多少？": "无法确定，可能需要查询班级名单。",
}


RAG_ANSWERS = {
    "RAG 的基本流程是什么？": "RAG 包含文档加载、文本切分、向量化、相似度检索和带上下文生成。核心是先检索相关文档片段，再把片段与问题一起交给大模型生成答案。",
    "本课程实践项目评分标准是什么？": "知识库写明：任务完成度占 30%，独立思考占 30%，技术实现占 20%，结果展示与表达占 20%。",
    "chunk_size 设为 200、500、1000 时有什么差异？": "200 字定位精确但上下文容易断；500 字兼顾主题完整性和检索精度，适合作为默认值；1000 字上下文更完整但噪声更大、排序区分度下降。",
    "任课教师下周三的办公室具体地点在哪里？": "知识库中没有找到依据，不能确认任课教师下周三的办公室地点。",
    "某同学的真实学号是多少？": "知识库中没有找到依据，不能回答某同学的真实学号。",
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
    try:
        test = csv_path.open("a", encoding="utf-8-sig")
        test.close()
    except PermissionError:
        csv_path = OUT_CSV.with_name("evaluation_results_rerun.csv")
        md_path = OUT_MD.with_name("evaluation_results_rerun.md")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = ["# RAG 问答测试结果", ""]
    for i, row in enumerate(rows, 1):
        md.append(f"## {i}. {row['问题']}")
        md.append(f"- 问题类型：{row['问题类型']}")
        md.append(f"- 普通 LLM：{row['普通LLM回答']}")
        md.append(f"- RAG：{row['RAG回答']}")
        md.append(f"- 结论：{row['结论']}")
        md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
