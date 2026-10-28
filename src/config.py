"""全局配置：统一从环境变量 / .env 文件读取，所有模块共享。"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # 未安装 python-dotenv 时静默跳过
    load_dotenv = None

# 项目根目录（src 的上一级）
ROOT_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------- LLM（OpenAI 兼容端点，预留 vLLM / Ollama / LM Studio 等） ----------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.1)

# ---------- Embedding ----------
# 优先使用项目内离线模型（offline/models/），不存在时回退到 HuggingFace 在线下载
_LOCAL_BGE = ROOT_DIR / "offline" / "models" / "bge-small-zh-v1.5"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(_LOCAL_BGE) if _LOCAL_BGE.exists() else "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto / cpu / cuda

# ---------- 向量库（Chroma 持久化） ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT_DIR / "data" / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_rag")

# ---------- 检索与文本分割 ----------
K = _get_int("K", 5)                     # 每次检索返回的文档块数量
CHUNK_SIZE = _get_int("CHUNK_SIZE", 512)  # 分割块大小（字符）
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 50)  # 块间重叠
NUM_MULTI_QUERIES = _get_int("NUM_MULTI_QUERIES", 3)  # MultiQuery 变体问题数

# 检索相似度门槛：低于该值的命中视为“无关”，直接丢弃。
# 目的：避免库里根本没有相关内容时，Chroma 仍强制返回 top-K 无关条文，
# 误导用户以为“原文已查到”，也让 LLM 拿到无关上下文。
# （实测：真实命中约 0.48+，库外/无关问题在改写变体下最高约 0.37，默认 0.40 可干净分隔）
RETRIEVAL_SCORE_THRESHOLD = _get_float("RETRIEVAL_SCORE_THRESHOLD", 0.40)

# ---------- 数据目录 ----------
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", str(ROOT_DIR / "test_samples"))
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
