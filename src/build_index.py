"""向量索引构建：文档 → 加载 → 分割 → BGE 嵌入 → Chroma 持久化 → retriever。

用法：
    python -m src.build_index --data test_samples            # 构建（增量）
    python -m src.build_index --data test_samples --rebuild  # 清空重建
"""
import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from src.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                        DEFAULT_DATA_DIR, K, RETRIEVAL_SCORE_THRESHOLD)
from src.embeddings import get_embeddings
from src.loader import load_documents
from src.splitter import split_documents


# 基于“正文 + 稳定元数据”的确定性 ID：同一内容重复添加时自动去重（upsert）。
# （langchain-chroma 0.2+ 在未提供 ids 时使用 uuid4，会导致重复添加产生副本）
def _chunk_id(chunk: Document) -> str:
    meta = chunk.metadata or {}
    stable = {k: v for k, v in meta.items()
              if isinstance(v, (str, int, float, bool))}
    key = chunk.page_content + json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _chunk_ids(chunks: List[Document]) -> List[str]:
    return [_chunk_id(c) for c in chunks]


def _create_vectorstore(embeddings, chroma_dir: str = None, collection_name: str = None):
    """创建/打开持久化 Chroma 客户端（兼容不同 langchain-chroma 版本）。"""
    chroma_dir = chroma_dir or CHROMA_DIR
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    kwargs = {
        "collection_name": collection_name or COLLECTION_NAME,
        "embedding_function": embeddings,
    }
    try:
        from langchain_chroma import Chroma
        params = inspect.signature(Chroma.__init__).parameters
        if "path" in params and "persist_directory" not in params:
            kwargs["path"] = chroma_dir      # langchain-chroma >= 0.2
        else:
            kwargs["persist_directory"] = chroma_dir  # langchain-chroma < 0.2
        return Chroma(**kwargs)
    except ImportError:
        from langchain_community.vectorstores import Chroma as ChromaCompat
        return ChromaCompat(persist_directory=chroma_dir, **kwargs)


def get_vectorstore(chroma_dir: str = None, collection_name: str = None, embeddings=None):
    """打开已有的持久化向量库（不存在则创建空集合）。"""
    return _create_vectorstore(embeddings or get_embeddings(), chroma_dir, collection_name)


def get_retriever(k: int = K, search_type: str = "similarity_score_threshold",
                  score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
                  embeddings=None):
    """构建基础 retriever。

    默认 search_type="similarity_score_threshold"：低于 score_threshold 的命中会被丢弃。
    这样当知识库中根本没有与问题相关的内容时，不会强行返回 top-K 无关条文，
    避免 LLM 拿到无关上下文后误答 / 误报“无法回答”，也避免界面把无关片段当成“来源”。
    如需纯相似度 Top-K，可传 search_type="similarity"。
    """
    vs = get_vectorstore(embeddings=embeddings)
    search_kwargs = {"k": k}
    if search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold
    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def build_index(
    data_dir: str,
    rebuild: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embeddings=None,
) -> dict:
    """加载目录文档 → 分割 → 写入 Chroma。

    Args:
        data_dir: 文档目录。
        rebuild: True 时清空向量库后重建；False 时增量添加（内容相同的块自动去重）。
        chunk_size / chunk_overlap: 分割参数。
        embeddings: 复用已有 embedding 实例。

    Returns:
        统计信息 dict：{status, docs, chunks, total_in_db}
    """
    embeddings = embeddings or get_embeddings()

    if rebuild:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        print(f"[build_index] 已清空向量库：{CHROMA_DIR}")

    docs = load_documents(data_dir)
    if not docs:
        return {"status": "no_docs", "docs": 0, "chunks": 0, "total_in_db": _count()}

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = _create_vectorstore(embeddings)
    vs.add_documents(chunks, ids=_chunk_ids(chunks))  # 确定性 ID，重复添加自动去重

    total = _count(embeddings=embeddings)
    print(f"[build_index] 写入完成：{len(chunks)} 个块，向量库现有 {total} 个块")
    return {
        "status": "built",
        "docs": len(docs),
        "chunks": len(chunks),
        "total_in_db": total,
    }


def add_documents(documents: List[Document], embeddings=None,
                  chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> int:
    """向已有索引增量添加文档（供 Streamlit 上传文件使用）。"""
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = get_vectorstore(embeddings=embeddings)
    vs.add_documents(chunks, ids=_chunk_ids(chunks))
    return len(chunks)


def _count(embeddings=None) -> int:
    try:
        vs = get_vectorstore(embeddings=embeddings)
        return vs._collection.count()
    except Exception:
        return 0


def index_stats(embeddings=None) -> int:
    """向量库当前文档块总数。"""
    return _count(embeddings=embeddings)


def _demo_retrieval(k: int = K):
    """构建索引后做一次检索冒烟测试，打印 Top-k 命中片段及相似度分数。"""
    print("\n" + "=" * 60)
    print("检索冒烟测试（无需 LLM）……")
    vs = get_vectorstore()
    question = "定金不得超过主合同标的额的多少比例？"
    docs = vs.similarity_search_with_relevance_scores(question, k=k)
    print(f"问题：{question}\n命中 {len(docs)} 个片段：")
    for i, (d, score) in enumerate(docs, 1):
        src = Path(d.metadata.get("source", "?")).name
        page = d.metadata.get("page")
        loc = f"第{page + 1}页" if isinstance(page, int) else "—"
        print(f"\n[{i}] 来源：{src}（{loc}） score={score:.3f}")
        print("    " + d.page_content[:120].replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="文档目录")
    parser.add_argument("--rebuild", action="store_true", help="清空并重建索引")
    parser.add_argument("--k", type=int, default=K, help="检索返回块数")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    stats = build_index(args.data, rebuild=args.rebuild,
                        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"[build_index] 结果：{stats}")
    _demo_retrieval(k=args.k)


if __name__ == "__main__":
    main()
