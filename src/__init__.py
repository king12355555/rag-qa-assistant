"""企业级 RAG 智能问答助手

模块结构：
- config.py       全局配置（环境变量 / .env）
- loader.py       文档加载（.pdf/.docx/.txt/.md）
- splitter.py     文本分割策略（512/50 + MarkdownHeader 保留表格结构）
- embeddings.py   BGE 中文 Embedding 模型（本地加载）
- llm.py          LLM 统一接口（OpenAI 兼容端点）
- build_index.py  向量索引构建（Chroma 持久化 + retriever）
- qa_chain.py     RAG 问答链（MultiQuery 并行检索 + 来源引用）
- app.py          Streamlit 界面
- cli.py          命令行问答
"""

__version__ = "1.0.0"
