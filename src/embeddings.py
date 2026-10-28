"""Embedding 模型：BAAI/bge-small-zh-v1.5（本地加载，无需联网推理）。

- 使用 HuggingFaceBgeEmbeddings 封装 sentence-transformers
- 自动选择 cuda / cpu（可通过 EMBEDDING_DEVICE 覆盖）
- 启用 BGE 官方中文检索指令前缀 + 向量归一化，提升检索质量
- 模型为本地目录时强制离线模式，避免联网重试拖慢启动
"""
import os
from functools import lru_cache
from pathlib import Path

from src.config import EMBEDDING_DEVICE, EMBEDDING_MODEL

# BGE 官方推荐的中文检索指令（仅用于 query，文档侧不追加）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def get_embeddings(device: str | None = None, model_name: str | None = None):
    """获取 Embedding 模型实例（进程内只加载一次）。

    Args:
        device: auto / cpu / cuda。默认取配置 EMBEDDING_DEVICE。
        model_name: 模型名称或本地路径，默认 BAAI/bge-small-zh-v1.5。
    """
    model_name = model_name or EMBEDDING_MODEL
    device = device or EMBEDDING_DEVICE

    # 本地模型目录：强制离线，禁止任何 huggingface 联网请求（SSL/断网时不再卡启动）
    if Path(model_name).is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[embeddings] 加载模型 {model_name}（device={device}），首次加载需要下载权重……")

    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # 余弦相似度
        query_instruction=BGE_QUERY_INSTRUCTION,
    )
