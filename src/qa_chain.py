"""检索增强生成链（RAG）：

1. ParallelMultiQueryRetriever（MultiQueryRetriever 增强版）：
   LLM 生成 N 个语义变体问题 → 线程池并行检索 → 按内容去重合并；
2. RetrievalQA.from_chain_type（stuff 模式）按 Prompt 模板生成回答；
3. answer_with_sources：回答同时返回来源（文件 + 页码/章节 + 片段摘要）。

Prompt 模板（可自定义）：
    你是一个专业助手。基于以下上下文信息回答问题。
    如果信息不足，请明确说"根据现有资料无法回答"。
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field

# 兼容 langchain 0.2 / 0.3 的包路径
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    from langchain_community.retrievers.multi_query import MultiQueryRetriever

from langchain.chains import LLMChain, RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.config import K, NUM_MULTI_QUERIES

# 回答 Prompt 模板
PROMPT_TEMPLATE = """你是一个专业助手。基于以下上下文信息回答问题。
如果信息不足，请明确说"根据现有资料无法回答"。

上下文：
{context}

问题：{question}

回答："""

# MultiQuery 查询改写 Prompt 模板
MULTI_QUERY_PROMPT_TEMPLATE = """你是一名专业的检索助理。请把下面的用户问题改写成 {num} 个语义相同但表达方式不同的变体问题，
以便从向量数据库中检索到更多样化的相关文档，弥补单一相似度检索的不足。
要求：每行只输出一个变体问题，不要编号，不要空行，不要输出任何其他内容。

原始问题：{question}

变体问题："""


class ParallelMultiQueryRetriever(MultiQueryRetriever):
    """MultiQueryRetriever 增强版。

    - LLM 生成 num_queries 个语义变体问题（含原始问题）；
    - 使用线程池对所有变体问题「并行」执行向量检索；
    - 对命中结果按正文内容「去重合并」，保证上下文无冗余。
    """

    num_queries: int = NUM_MULTI_QUERIES
    max_workers: int = 4
    last_queries: List[str] = Field(default_factory=list, exclude=True)
    last_docs: List[Document] = Field(default_factory=list, exclude=True)

    @classmethod
    def from_llm(
        cls,
        retriever,
        llm,
        num_queries: int = NUM_MULTI_QUERIES,
        include_original: bool = True,
        max_workers: Optional[int] = None,
        prompt: Optional[PromptTemplate] = None,
    ) -> "ParallelMultiQueryRetriever":
        prompt = prompt or PromptTemplate(
            template=MULTI_QUERY_PROMPT_TEMPLATE,
            input_variables=["question", "num"],
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            include_original=include_original,
            unique=True,  # 旧版 langchain 支持，新版会忽略
            num_queries=num_queries,
            max_workers=max_workers or max(4, num_queries),
        )

    # ------------------------------------------------------------------
    def generate_queries(self, question: str) -> List[str]:
        """调用 LLM 生成 num_queries 个变体问题；LLM 不可用时退回原始问题。"""
        raw = ""
        try:
            result = self.llm_chain.invoke({"question": question, "num": self.num_queries})
            raw = result.get("text", "") or ""
        except Exception as exc:
            print(f"[MultiQueryRetriever] 查询改写失败，退回原始问题：{exc}")

        queries: List[str] = []
        for line in raw.splitlines():
            line = line.strip().strip('"\'“”‘’「」')
            # 去掉 "1. " / "1、" / "一、" 之类的序号前缀
            line = re.sub(r"^\s*(?:第?\s*[0-9一二三四五六七八九十百]+[.、）)]\s*)", "", line)
            if line and line not in queries:
                queries.append(line)
        queries = queries[: self.num_queries]
        if self.include_original and question not in queries:
            queries.insert(0, question)  # 原始问题优先
        return queries or [question]

    def _get_relevant_documents(self, question: str) -> List[Document]:
        """变体问题 → 并行检索 → 去重合并。"""
        queries = self.generate_queries(question)
        self.last_queries = list(queries)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            docs_list = list(pool.map(lambda q: self.retriever.invoke(q), queries))
        merged = [d for docs in docs_list for d in docs]
        unique = self._dedupe(merged)
        self.last_docs = list(unique)  # 记录本次实际命中的文档（供回答后判断是否“无命中”）
        return unique

    @staticmethod
    def _dedupe(documents: List[Document]) -> List[Document]:
        """按正文内容去重，保留首次出现顺序。"""
        seen, unique = set(), []
        for doc in documents:
            key = doc.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique


# ---------------------------------------------------------------------------
# QA 链
# ---------------------------------------------------------------------------

def build_qa_chain(
    llm,
    retriever,
    template: str = PROMPT_TEMPLATE,
    chain_type: str = "stuff",
) -> RetrievalQA:
    """构建 RetrievalQA 问答链（stuff 模式 + 自定义 Prompt + 返回来源文档）。"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def create_qa_system(
    k: int = K,
    num_queries: int = NUM_MULTI_QUERIES,
    llm=None,
    base_retriever=None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    """一键装配：Embedding + Chroma → retriever → MultiQuery 并行检索 → QA 链。

    Returns:
        {"chain": RetrievalQA, "retriever": ParallelMultiQueryRetriever, "llm": llm}
    """
    from src.build_index import get_retriever
    from src.llm import get_llm

    llm = llm or get_llm(base_url=base_url, api_key=api_key, model=model,
                         temperature=temperature)
    base = base_retriever or get_retriever(k=k)
    multi = ParallelMultiQueryRetriever.from_llm(base, llm, num_queries=num_queries)
    chain = build_qa_chain(llm, multi)
    return {"chain": chain, "retriever": multi, "llm": llm}


# ---------------------------------------------------------------------------
# 来源引用
# ---------------------------------------------------------------------------

def format_sources(source_documents: List[Document]) -> List[dict]:
    """从检索命中的文档块中提取来源信息（文件 + 页码/章节），按来源去重。

    Returns:
        [{"source": 文件名, "location": 页码或章节, "excerpt": 片段摘要}, ...]
    """
    seen: Dict[tuple, dict] = {}
    for doc in source_documents:
        meta = doc.metadata or {}
        source = Path(str(meta.get("source", "未知来源"))).name
        page = meta.get("page")
        headers = [meta[k] for k in ("H1", "H2", "H3", "H4", "H5", "H6") if meta.get(k)]
        if isinstance(page, int):
            location = f"第 {page + 1} 页"
        elif headers:
            location = " > ".join(headers)
        else:
            location = "全文"
        key = (source, location)
        if key not in seen:
            seen[key] = {
                "source": source,
                "location": location,
                "excerpt": doc.page_content[:150].replace("\n", " "),
            }
    return list(seen.values())


NO_HIT_MESSAGE = ("知识库中未检索到与您的问题相关的内容（候选条文相似度均低于阈值）。\n\n"
                  "可能原因与建议：\n"
                  "1. 知识库尚未收录该问题对应的法律条文（例如询问故意杀人需《刑法》，"
                  "而当前库中只有《民法典》等内容），请把文档目录指向含相关法条的文件夹后\n"
                  "   点击「构建/更新索引」，或直接上传文档（不要只上传到 test_samples 而不重建索引）；\n"
                  "2. 换个更明确、含关键词的说法再试一次。")


def answer_with_sources(
    question: str,
    qa_chain: RetrievalQA,
    retriever: Optional[ParallelMultiQueryRetriever] = None,
) -> dict:
    """执行问答并返回结构化结果。

    当相似度阈值把检索结果全部过滤掉（知识库里确实没有相关内容）时，
    不再把 LLM 那句笼统的“无法回答”直接抛给用户，而是返回明确的引导说明。

    Returns:
        {"question", "answer", "queries"(改写的问题), "sources": [...], "no_hit": bool}
    """
    result = qa_chain.invoke({"query": question})
    docs = result.get("source_documents", []) or list(
        getattr(retriever, "last_docs", []) or []
    )
    queries = list(getattr(retriever, "last_queries", []) or [])
    if not docs:
        return {
            "question": question,
            "answer": NO_HIT_MESSAGE,
            "queries": queries,
            "sources": [],
            "no_hit": True,
        }
    return {
        "question": question,
        "answer": result.get("result", ""),
        "queries": queries,
        "sources": format_sources(docs),
        "no_hit": False,
    }
