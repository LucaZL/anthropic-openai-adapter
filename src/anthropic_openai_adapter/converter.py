"""
协议转换核心逻辑

Anthropic Messages API ↔ OpenAI Chat Completions API 双向转换。
无外部依赖，纯 Python 标准库。
"""

import json
from typing import Dict, Generator, List, Optional


def anthropic_to_openai_request(body: dict, model_map: Optional[Dict[str, str]] = None) -> dict:
    """
    将 Anthropic Messages 请求体转为 OpenAI Chat Completions 格式。

    Args:
        body: Anthropic /v1/messages 请求体
        model_map: 模型名映射表，如 {"claude-sonnet-4-5": "gpt-4o"}。
                   未命中时保留原模型名。

    Returns:
        OpenAI /v1/chat/completions 请求体
    """
    messages = []

    # system prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [b.get("text", "") for b in system if b.get("type") == "text"]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    # 先输出前面积累的文本
                    if text_parts:
                        messages.append({"role": role, "content": "\n".join(text_parts)})
                        text_parts = []
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }],
                    })
                elif block.get("type") == "tool_result":
                    if text_parts:
                        messages.append({"role": role, "content": "\n".join(text_parts)})
                        text_parts = []
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b.get("text", "") for b in tool_content if b.get("type") == "text"
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tool_content),
                    })
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

    # model mapping
    model = body.get("model", "")
    if model_map:
        model = model_map.get(model, model)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
    }

    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]

    # Anthropic tools → OpenAI tools
    anthropic_tools = body.get("tools")
    if anthropic_tools:
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        payload["tools"] = openai_tools

    return payload


def openai_to_anthropic_response(
    openai_resp: dict,
    model: str = "",
    use_reasoning: bool = False,
) -> dict:
    """
    将 OpenAI Chat Completions 响应转为 Anthropic Messages 格式。

    Args:
        openai_resp: OpenAI 响应体
        model: 返回给客户端的模型名
        use_reasoning: 是否将 reasoning 字段也作为内容输出。
                       默认 False，只取 content（避免思考过程污染对话历史）。

    Returns:
        Anthropic Messages 响应体
    """
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    content_text = message.get("content") or ""
    reasoning_text = message.get("reasoning") or "" if use_reasoning else ""

    content = []
    if reasoning_text:
        content.append({"type": "text", "text": f"<thinking>\n{reasoning_text}\n</thinking>"})
    if content_text:
        content.append({"type": "text", "text": content_text})

    # OpenAI tool_calls → Anthropic tool_use
    for tc in message.get("tool_calls") or []:
        func = tc.get("function", {})
        try:
            tool_input = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {"raw": func.get("arguments", "")}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{id(tc)}"),
            "name": func.get("name", ""),
            "input": tool_input,
        })

    usage = openai_resp.get("usage", {})
    stop_reason = "end_turn"
    finish = choice.get("finish_reason", "")
    if finish == "length":
        stop_reason = "max_tokens"
    elif finish == "tool_calls":
        stop_reason = "tool_use"

    resp_model = model or openai_resp.get("model", "")

    return {
        "id": openai_resp.get("id", "msg_adapter"),
        "type": "message",
        "role": "assistant",
        "model": resp_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_stream_to_anthropic_stream(
    openai_lines: List[str],
    model: str = "",
    msg_id: str = "msg_adapter",
    use_reasoning: bool = False,
) -> Generator[str, None, None]:
    """
    将 OpenAI SSE 流（行列表）转成 Anthropic SSE 流（逐行 yield）。

    Args:
        openai_lines: OpenAI SSE 原始行列表
        model: 模型名
        msg_id: 消息 ID
        use_reasoning: 是否输出 reasoning

    Yields:
        Anthropic SSE 格式字符串（event: xxx\\ndata: {...}\\n\\n）
    """
    # message_start
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    stop_reason = "end_turn"
    total_text = ""
    tool_calls_acc: Dict[int, dict] = {}
    text_block_started = False

    for line in openai_lines:
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        finish = chunk.get("choices", [{}])[0].get("finish_reason")

        # 文本 content
        text = delta.get("content") or ""
        if text:
            if not text_block_started:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                text_block_started = True
            total_text += text
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })

        # tool_calls delta
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
            if tc.get("id"):
                tool_calls_acc[idx]["id"] = tc["id"]
            func = tc.get("function", {})
            if func.get("name"):
                tool_calls_acc[idx]["name"] = func["name"]
            if func.get("arguments"):
                tool_calls_acc[idx]["arguments"] += func["arguments"]

        if finish == "length":
            stop_reason = "max_tokens"
        elif finish == "stop":
            stop_reason = "end_turn"
        elif finish == "tool_calls":
            stop_reason = "tool_use"

    # 关闭文本块
    if text_block_started:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    # 空响应保底
    if not text_block_started and not tool_calls_acc:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    # tool_use blocks
    block_index = 1 if text_block_started else 0
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            tool_input = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            tool_input = {"raw": tc["arguments"]}

        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tc["id"] or f"toolu_stream_{idx}",
                "name": tc["name"],
                "input": {},
            },
        })
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(tool_input),
            },
        })
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        block_index += 1

    # message_delta
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": len(total_text)},
    })

    # message_stop
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict) -> str:
    """格式化 SSE 行"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
