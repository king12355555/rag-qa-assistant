"""Streamlit 界面：文档上传 / 索引管理 / 多轮问答 / 来源引用。

启动：
    streamlit run src/app.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.build_index import add_documents, build_index, index_stats
from src.config import (CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DEFAULT_DATA_DIR,
                        EMBEDDING_MODEL, K, NUM_MULTI_QUERIES, ROOT_DIR, UPLOAD_DIR)
from src.loader import load_documents
from src.llm import check_llm_health, get_llm
from src.qa_chain import answer_with_sources, create_qa_system

st.set_page_config(page_title="企业级 RAG 智能问答助手", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 对话历史 [{"role": ..., "content": ...}]


@st.cache_resource(show_spinner="正在加载 RAG 系统（首次加载 Embedding 模型较慢）……")
def get_qa_system(base_url: str, api_key: str, model: str, temperature: float,
                  k: int, num_queries: int):
    """缓存 QA 系统实例；配置变化或重建索引后自动重建。"""
    return create_qa_system(
        k=k, num_queries=num_queries,
        base_url=base_url, api_key=api_key, model=model, temperature=temperature,
    )


def reset_qa_system() -> None:
    """重建索引 / 配置变化后清空缓存。"""
    st.cache_resource.clear()
    st.session_state.pop("qa_ready", None)


def save_llm_config_to_env(base_url: str, api_key: str, model: str, temperature: float) -> str:
    """把界面填写的 LLM 配置持久化写入 .env，下次启动自动生效。"""
    env_path = ROOT_DIR / ".env"
    updates = {
        "LLM_BASE_URL": base_url.strip(),
        "LLM_API_KEY": api_key.strip(),
        "LLM_MODEL": model.strip(),
        "LLM_TEMPERATURE": f"{temperature:g}",
    }

    def _fmt(key: str, value: str) -> str:
        # 值含空格 / 井号 / 引号时用双引号包裹，避免 .env 解析错误
        if value and any(c in value for c in ' \t#"\'\\'):
            return f'{key}="{value}"'
        return f"{key}={value}"

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        done = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                lines[i] = _fmt(key, updates[key])
                done.add(key)
        for key, value in updates.items():  # 追加 .env 中没有的键
            if key not in done:
                lines.append(_fmt(key, value))
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(
            "\n".join(_fmt(k, v) for k, v in updates.items()) + "\n",
            encoding="utf-8",
        )
    # 同步到当前进程，保证本次运行也立即生效
    for key, value in updates.items():
        os.environ[key] = value
    return "✅ 已保存到 .env，下次启动自动生效（本次运行也已生效）"


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 配置")

    with st.expander("🤖 LLM（OpenAI 兼容端点）", expanded=True):
        base_url = st.text_input("Base URL", value=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
        api_key = st.text_input("API Key", value=os.getenv("LLM_API_KEY", "EMPTY"), type="password")
        model = st.text_input("模型名", value=os.getenv("LLM_MODEL", "qwen2.5-7b-instruct"))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🧪 测试连接", use_container_width=True):
                with st.spinner("正在测试……"):
                    st.info(check_llm_health(get_llm(base_url, api_key, model, temperature)))
        with col_btn2:
            if st.button("💾 保存配置", use_container_width=True):
                st.success(save_llm_config_to_env(base_url, api_key, model, temperature))

    with st.expander("📑 索引管理", expanded=True):
        data_dir = st.text_input("文档目录", value=DEFAULT_DATA_DIR)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 构建/更新索引", use_container_width=True):
                with st.spinner("正在加载文档并构建向量索引……"):
                    stats = build_index(data_dir, rebuild=False)
                reset_qa_system()
                st.success(f"完成：{stats['docs']} 个文档 → {stats['chunks']} 个块，"
                           f"向量库共 {stats['total_in_db']} 个块")
                st.rerun()
        with col2:
            if st.button("🗑️ 清空重建", use_container_width=True):
                with st.spinner("正在清空并重建……"):
                    stats = build_index(data_dir, rebuild=True)
                reset_qa_system()
                st.success(f"重建完成：向量库共 {stats['total_in_db']} 个块")
                st.rerun()

        uploaded = st.file_uploader(
            "📤 上传文档（加入索引）",
            type=["pdf", "docx", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )
        if uploaded:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved = []
            for f in uploaded:
                target = UPLOAD_DIR / f.name
                target.write_bytes(f.getbuffer())
                saved.append(str(target))
            with st.spinner("正在解析并写入向量库……"):
                n = add_documents(load_documents(str(UPLOAD_DIR)))
            reset_qa_system()
            st.success(f"已上传 {len(saved)} 个文件，新增 {n} 个块到向量库")
            st.rerun()

        with st.spinner("正在读取向量库……"):
            total = index_stats()
        st.caption(f"向量库（{COLLECTION_NAME}）当前块数：**{total}**")

    with st.expander("🔧 检索参数"):
        k = st.slider("检索返回块数 K", 1, 20, K)
        num_queries = st.slider("MultiQuery 变体问题数", 1, 5, NUM_MULTI_QUERIES)
        st.caption(f"分割参数：chunk_size={CHUNK_SIZE}，overlap={CHUNK_OVERLAP}")

    st.divider()
    st.caption(f"Embedding：{EMBEDDING_MODEL}")

# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------
st.title("📚 企业级 RAG 智能问答助手")
st.caption("本地向量检索（BGE + Chroma）｜多查询并行检索｜来源引用｜OpenAI 兼容 LLM")

# 尚未开始对话时的使用引导
if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("#### 👋 开始使用")
        st.markdown(
            "1. 在**页面底部输入框**输入问题后按回车即可提问（如：*定金不得超过主合同标的额的多少比例？*）；\n"
            "2. 若回答失败，先在左侧「🤖 LLM」点 **🧪 测试连接**，确认 LLM 端点可达；\n"
            "3. 左侧填写的 LLM 配置可点 **💾 保存配置** 写入 `.env`，下次启动自动生效。"
        )
        st.markdown(
            f"当前 LLM 配置：**{model}** @ `{base_url}` ｜ "
            f"向量库块数：**{index_stats()}** ｜ Embedding：`{EMBEDDING_MODEL}`"
        )

# 对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if question := st.chat_input("请输入您的问题（例如：定金不得超过主合同标的额的多少比例？）"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        qa = get_qa_system(base_url, api_key, model, temperature, k, num_queries)
        with st.chat_message("assistant"):
            with st.spinner("检索中……"):
                payload = answer_with_sources(question, qa["chain"], qa["retriever"])
            st.markdown(payload["answer"])

            with st.expander(f"📎 来源引用（{len(payload['sources'])} 处）"):
                for i, s in enumerate(payload["sources"], 1):
                    st.markdown(
                        f"**{i}. {s['source']}** ｜ `{s['location']}`\n\n"
                        f"> {s['excerpt']}…"
                    )
            if payload.get("queries"):
                st.caption("🔍 MultiQuery 改写问题：" + " ｜ ".join(payload["queries"]))

        st.session_state.messages.append(
            {"role": "assistant", "content": payload["answer"]}
        )
    except Exception as exc:
        st.error(f"问答失败：{exc}")
        st.info("请确认：① LLM 端点已启动且 Base URL 正确；② 向量库已构建（左侧索引管理）。")
