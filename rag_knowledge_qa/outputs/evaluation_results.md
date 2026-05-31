# RAG 问答测试结果

## 1. RAG 的基本流程是什么？
- 问题类型：文档内
- 普通 LLM：RAG 通常是先检索资料，再让大模型结合资料回答问题。
- RAG：根据公开资料整理，RAG 的流程包括资料加载、文本切分、向量化、向量数据库保存、相似度检索，以及把检索片段和问题一起交给聊天模型生成答案。来源包括 LangChain RAG 教程和检索概念文档。
- 检索证据摘要：LangChain 的 RAG 教程把一个典型问答系统拆成几个步骤：先加载文本资料，再把资料拆分成较小片段，然后使用向量模型把片段转成 embedding。
- 结论：RAG 更具体且可追溯

## 2. LangChain RAG 流程中为什么要进行文本切分？
- 问题类型：文档内
- 普通 LLM：文本切分通常是为了让内容更适合检索和放入模型上下文。
- RAG：文本切分是因为长文档不适合直接检索，也可能超过模型上下文窗口。把资料拆成较小 chunks 后，检索器更容易返回主题清晰的片段；chunk_overlap 还能减少边界信息丢失。
- 检索证据摘要：LangChain 的 RAG 教程通常会使用 TextLoader、WebBaseLoader 或其他 loader 读取资料，再使用 RecursiveCharacterTextSplitter 进行切分。
- 结论：RAG 更具体且可追溯

## 3. Chroma collection 通常保存哪些内容？
- 问题类型：文档内
- 普通 LLM：Chroma collection 大概用于保存向量和文档，具体字段需要看文档。
- RAG：Chroma 文档中 collection 用来组织知识库记录。每条记录通常包含 id、document、embedding 和 metadata；document 保存文本片段，embedding 保存向量，metadata 保存来源、页码、章节等辅助信息。
- 检索证据摘要：Chroma 使用 collection 组织数据。一个 collection 可以理解为一个知识库集合，里面包含多条记录。
- 结论：RAG 更具体且可追溯

## 4. 任课教师下周三的办公室具体地点在哪里？
- 问题类型：文档外
- 普通 LLM：可能在教学楼办公室，建议查看课程通知。
- RAG：知识库中没有找到依据。当前知识库只整理公开的 RAG、LangChain、Chroma、DeepSeek 和 Streamlit 资料，没有教师办公室地点。
- 检索证据摘要：无相关公开资料证据。
- 结论：RAG 能识别知识边界

## 5. 某同学的真实学号是多少？
- 问题类型：文档外
- 普通 LLM：无法确定，可能需要查询班级名单。
- RAG：知识库中没有找到依据。公开资料整理中不包含任何同学真实学号，也不应该编造个人隐私信息。
- 检索证据摘要：无相关公开资料证据。
- 结论：RAG 能识别知识边界
