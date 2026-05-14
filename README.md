# anthropic-openai-adapter

Anthropic Messages API → OpenAI Chat Completions API 反向代理。

将发往 `/v1/messages`（Anthropic 协议）的请求无缝转换为 `/v1/chat/completions`（OpenAI 协议）转发到任意兼容后端（OpenAI、vLLM、Ollama、GLM 等），并将响应转换回 Anthropic 格式。

**零外部依赖**，仅使用 Python 标准库。

## 功能

- Anthropic → OpenAI 请求转换（system、messages、tools）
- OpenAI → Anthropic 响应转换（非流式 + SSE 流式）
- `tool_use` ↔ `tool_calls` 双向转换
- 模型名映射（如 `claude-sonnet-4-5` → `gpt-4o`）
- 可选 `reasoning` 字段透传（用于推理模型）
- 多线程并发处理
- 支持 CLI 独立运行或作为 Python 库嵌入

## 安装

```bash
pip install anthropic-openai-adapter
```

## 使用

### CLI 独立运行

```bash
# 代理到本地 Ollama
anthropic-openai-adapter --backend-url http://localhost:11434 --port 8081

# 代理到 OpenAI，带模型映射
anthropic-openai-adapter \
  --backend-url https://api.openai.com \
  --api-key sk-xxx \
  --model-map '{"claude-sonnet-4-5": "gpt-4o", "claude-haiku-3-5": "gpt-4o-mini"}'

# 代理到 vLLM / GLM 等
anthropic-openai-adapter \
  --backend-url http://glm-server:8000 \
  --model-map '{"claude-sonnet-4-5": "glm-5.1"}'
```

### 作为 Python 库

```python
from anthropic_openai_adapter import start_server

# 在后台线程启动
server = start_server(
    backend_url="http://localhost:11434",
    api_key="your-key",
    port=8081,
    model_map={"claude-sonnet-4-5": "qwen2.5:72b"},
)

# 现在可以将 Claude Code CLI 指向 http://127.0.0.1:8081
# export ANTHROPIC_BASE_URL=http://127.0.0.1:8081
```

### 仅使用转换函数

```python
from anthropic_openai_adapter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)

# 转换请求
openai_req = anthropic_to_openai_request(
    anthropic_body,
    model_map={"claude-sonnet-4-5": "gpt-4o"},
)

# 转换响应
anthropic_resp = openai_to_anthropic_response(openai_resp, model="claude-sonnet-4-5")
```

## 典型用例

1. **Claude Code CLI + 非 Anthropic 后端**：让 Claude Code 连接到 OpenAI、GLM、Qwen、Ollama 等
2. **统一 API 网关**：内部只维护 OpenAI 协议后端，对外同时支持 Anthropic 客户端
3. **开发测试**：在 Anthropic 客户端和 OpenAI 后端之间做协议桥接

## 协议映射

| Anthropic | OpenAI |
|-----------|--------|
| `system` (string/list) | `messages[0].role=system` |
| `messages[].content[].type=text` | `messages[].content` (string) |
| `messages[].content[].type=tool_use` | `messages[].tool_calls[]` |
| `messages[].content[].type=tool_result` | `messages[].role=tool` |
| `tools[].input_schema` | `tools[].function.parameters` |
| SSE `content_block_delta` | SSE `choices[].delta.content` |
| `stop_reason=tool_use` | `finish_reason=tool_calls` |

## 配置

| 参数 | 环境变量 | 说明 |
|------|----------|------|
| `--backend-url` | - | OpenAI 兼容后端地址 |
| `--api-key` | `ADAPTER_API_KEY` | 后端 API Key |
| `--host` | - | 监听地址（默认 127.0.0.1） |
| `--port` | - | 监听端口（默认 8081） |
| `--model-map` | - | 模型名映射 JSON |
| `--use-reasoning` | - | 透传 reasoning 字段 |

## License

MIT
