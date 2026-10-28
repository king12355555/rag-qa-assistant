"""LLM 统一接口：OpenAI 兼容端点。

只需把 base_url 指向任意 OpenAI 兼容服务即可无缝切换后端：
- vLLM        http://localhost:8000/v1
- Ollama      http://localhost:11434/v1
- LM Studio   http://localhost:1234/v1
- 自建网关 / 云厂商兼容网关

配置项见 .env（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_TEMPERATURE）。
"""
from typing import Optional

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE


def get_llm(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
):
    """构建 ChatOpenAI 兼容客户端。

    Args:
        base_url: OpenAI 兼容端点，如 http://localhost:8000/v1
        api_key: 密钥（本地服务通常填 EMPTY）
        model: 模型名，如 qwen2.5-7b-instruct / llama3.1-8b / gpt-4o-mini
        temperature: 采样温度，默认 0.1（偏向确定性回答）
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=base_url or LLM_BASE_URL,
        api_key=api_key or LLM_API_KEY,
        model=model or LLM_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        timeout=kwargs.pop("timeout", 120),
        max_retries=kwargs.pop("max_retries", 2),
        **kwargs,
    )


def check_llm_health(llm) -> str:
    """连通性自检：让模型回显一句话。"""
    try:
        resp = llm.invoke("请回复'连接正常'四个字。")
        text = resp.content if hasattr(resp, "content") else str(resp)
        return f"✅ LLM 连接正常：{text.strip()[:80]}"
    except Exception as exc:
        return f"❌ LLM 连接失败：{exc}"
