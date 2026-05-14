"""
anthropic-openai-adapter: Anthropic Messages API ↔ OpenAI Chat Completions 协议转换代理

将 Anthropic /v1/messages 请求转换为 OpenAI /v1/chat/completions 请求，
支持流式/非流式、tool_use/tool_calls 双向转换。

可用作：
  - 独立反向代理服务（CLI 启动）
  - Python 库嵌入到其他项目中
"""

from anthropic_openai_adapter.server import create_server, start_server
from anthropic_openai_adapter.converter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_stream_to_anthropic_stream,
)

__version__ = "0.1.0"

__all__ = [
    "create_server",
    "start_server",
    "anthropic_to_openai_request",
    "openai_to_anthropic_response",
    "openai_stream_to_anthropic_stream",
]
