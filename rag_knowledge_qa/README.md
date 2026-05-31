# RAG 知识库问答助手

本项目是《人工智能通识课》期末实践项目第六类：RAG 知识库问答。按照“所有数据都要联网搜索，不能自己创造”的要求，知识库内容改为公开网页资料整理版，并在 `data/course_ai_notes.txt` 中列出来源 URL。

GitHub 仓库：https://github.com/weizhigao9-dev/-01

## 项目结构

- `data/course_ai_notes.txt`：8000 字以上公开资料整理知识库，来源包括 LangChain、Chroma、DeepSeek、Streamlit 和 RAG 论文
- `src/rag_pipeline.py`：LangChain RAG 主流程
- `src/offline_demo.py`：无 API key 时可运行的离线演示
- `src/app.py`：Streamlit 简易 Web 界面
- `outputs/evaluation_results.csv`：5 个测试问题结果
- `outputs/evaluation_results.md`：普通 LLM 与 RAG 对比结论
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

## 数据来源说明

知识库正文不是凭空编写，而是基于公开资料整理，主要来源包括：

- https://python.langchain.com/docs/tutorials/rag/
- https://python.langchain.com/docs/concepts/retrieval/
- https://docs.trychroma.com/docs/overview/introduction
- https://docs.trychroma.com/docs/collections
- https://docs.trychroma.com/docs/embeddings/embedding-functions
- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/api/create-chat-completion
- https://docs.streamlit.io/develop/api-reference/widgets/st.text_input
- https://docs.streamlit.io/develop/api-reference/layout/st.sidebar
- https://arxiv.org/abs/2005.11401
