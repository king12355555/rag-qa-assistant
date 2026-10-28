"""文本分割策略：
- 常规文档 → RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
- 包含表格 / 标题结构的文档 → MarkdownHeaderTextSplitter 保留标题层级，
  再叠加 RecursiveCharacterTextSplitter 控制块大小（512/50）。

分割后的每个块继承原始 metadata（source / page 等），保证检索后能溯源。
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 针对中文优化分隔符，尽量避免在句中被切断
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ("####", "H4"),
    ("#####", "H5"),
    ("######", "H6"),
]


def _recursive_splitter(chunk_size: int = 512, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def _header_splitter() -> MarkdownHeaderTextSplitter:
    # strip_headers=False：标题保留在正文中，LLM 能看到章节上下文
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)


def split_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """根据文档特征自动选择分割策略，返回可直接入库的文档块。

    Args:
        documents: loader 返回的原始文档。
        chunk_size: 块大小（字符）。
        chunk_overlap: 块间重叠。

    Returns:
        分割后的文档块列表（metadata 已合并）。
    """
    recursive = _recursive_splitter(chunk_size, chunk_overlap)
    header_splitter = _header_splitter()
    size_splitter = _recursive_splitter(chunk_size, chunk_overlap)

    chunks: List[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        # 1) 包含表格的文档（docx 表格 / markdown 表格 / txt 表格）
        # 2) 带标题结构的 Markdown 文档
        use_markdown = bool(meta.get("has_table")) or (
            meta.get("is_markdown") and _has_headers(doc.page_content)
        )
        if use_markdown:
            chunks.extend(_split_with_headers(doc, header_splitter, size_splitter))
        else:
            chunks.extend(recursive.split_documents([doc]))

    print(f"[splitter] 分割完成：{len(documents)} 个文档 → {len(chunks)} 个块"
          f"（chunk_size={chunk_size}, overlap={chunk_overlap}）")
    return chunks


def _split_with_headers(doc: Document, header_splitter, size_splitter) -> List[Document]:
    """MarkdownHeaderTextSplitter 先按标题切分（保留层级元数据），再按大小二次切分。"""
    base_meta = {k: v for k, v in (doc.metadata or {}).items()
                 if k not in ("H1", "H2", "H3", "H4", "H5", "H6")}
    try:
        sections = header_splitter.split_text(doc.page_content)
    except Exception as exc:  # 文本不是合法 Markdown 时降级
        print(f"[splitter] 标题切分失败，降级为普通切分：{exc}")
        sections = [Document(page_content=doc.page_content, metadata={})]

    chunks: List[Document] = []
    for sec in sections:
        merged_meta = {**base_meta, **sec.metadata}
        sub = Document(page_content=sec.page_content, metadata=merged_meta)
        chunks.extend(size_splitter.split_documents([sub]))
    return chunks


def _has_headers(text: str) -> bool:
    import re
    return bool(re.search(r"^#{1,6}\s+\S", text, re.M))
