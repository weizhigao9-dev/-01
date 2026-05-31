# RAG 知识库问答助手

本项目是《人工智能通识课》期末实践项目第六类：RAG 知识库问答。目标是用 LangChain 搭建一个可更新、可追溯的课程资料问答系统，并对比普通 LLM 与 RAG 的回答差异。

GitHub 仓库：https://github.com/weizhigao9-dev/-01

## 项目结构

- `data/course_ai_notes.txt`：8000 字以上知识库原文
- `src/rag_pipeline.py`：LangChain RAG 主流程
- `src/offline_demo.py`：无 API key 时可运行的离线演示
- `src/app.py`：Streamlit 简易 Web 界面
- `outputs/evaluation_results.csv`：5 个测试问题结果
- `outputs/evaluation_results.md`：测试结论
- `docs/RAG知识库问答项目报告.docx`：期末报告
- `docs/RAG知识库问答展示PPT.pptx`：5 分钟展示 PPT

## 运行方式

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/offline_demo.py
```

如果要运行完整 LangChain + DeepSeek 版本，请设置：

```bash
set DEEPSEEK_API_KEY=你的 API Key
python src/rag_pipeline.py
```

展示 Web 页面：

```bash
streamlit run src/app.py
```

## 说明

由于本地环境未安装 LangChain，本仓库同时保留离线演示脚本，用于证明检索、问答对比和失败分析流程。正式提交前建议在联网环境安装依赖并运行 `src/rag_pipeline.py`。
