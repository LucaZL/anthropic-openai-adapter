"""Tests for anthropic_openai_adapter converter"""

import json
from anthropic_openai_adapter.converter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_stream_to_anthropic_stream,
)


def test_basic_message_conversion():
    """基本消息转换"""
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
        ],
    }
    result = anthropic_to_openai_request(body)
    assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert result["messages"][1] == {"role": "user", "content": "Hello"}
    assert result["max_tokens"] == 1024
    assert result["model"] == "claude-sonnet-4-5"


def test_model_mapping():
    """模型映射"""
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    result = anthropic_to_openai_request(body, model_map={"claude-sonnet-4-5": "gpt-4o"})
    assert result["model"] == "gpt-4o"


def test_model_mapping_no_match():
    """未命中模型映射时保留原名"""
    body = {
        "model": "claude-opus-4",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    result = anthropic_to_openai_request(body, model_map={"claude-sonnet-4-5": "gpt-4o"})
    assert result["model"] == "claude-opus-4"


def test_system_as_list():
    """system 为列表格式"""
    body = {
        "model": "test",
        "system": [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    result = anthropic_to_openai_request(body)
    assert result["messages"][0] == {"role": "system", "content": "Part 1\nPart 2"}


def test_tool_use_conversion():
    """tool_use content block → OpenAI tool_calls"""
    body = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "Beijing"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny, 25°C"},
            ]},
        ],
    }
    result = anthropic_to_openai_request(body)
    # assistant text
    assert result["messages"][1] == {"role": "assistant", "content": "Let me check."}
    # tool call
    tc_msg = result["messages"][2]
    assert tc_msg["role"] == "assistant"
    assert tc_msg["tool_calls"][0]["id"] == "toolu_123"
    assert tc_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    # tool result
    tool_msg = result["messages"][3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "toolu_123"
    assert tool_msg["content"] == "Sunny, 25°C"


def test_tools_schema_conversion():
    """Anthropic tools → OpenAI tools"""
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    result = anthropic_to_openai_request(body)
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"] == "get_weather"
    assert result["tools"][0]["function"]["parameters"]["type"] == "object"


def test_openai_to_anthropic_response():
    """OpenAI 响应 → Anthropic 格式"""
    openai_resp = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    result = openai_to_anthropic_response(openai_resp, model="claude-sonnet-4-5")
    assert result["type"] == "message"
    assert result["model"] == "claude-sonnet-4-5"
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "Hello!"
    assert result["stop_reason"] == "end_turn"
    assert result["usage"]["input_tokens"] == 10


def test_openai_tool_calls_to_anthropic():
    """OpenAI tool_calls → Anthropic tool_use"""
    openai_resp = {
        "id": "chatcmpl-456",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }
    result = openai_to_anthropic_response(openai_resp, model="test")
    assert result["stop_reason"] == "tool_use"
    assert result["content"][0]["type"] == "tool_use"
    assert result["content"][0]["name"] == "search"
    assert result["content"][0]["input"] == {"query": "test"}


def test_openai_reasoning_ignored_by_default():
    """reasoning 字段默认不输出"""
    openai_resp = {
        "id": "chatcmpl-789",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning": "Let me think step by step...",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    result = openai_to_anthropic_response(openai_resp, model="test", use_reasoning=False)
    assert len(result["content"]) == 1
    assert "thinking" not in result["content"][0]["text"]


def test_openai_reasoning_included_when_enabled():
    """use_reasoning=True 时输出 reasoning"""
    openai_resp = {
        "id": "chatcmpl-789",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "42",
                "reasoning": "Deep thought",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }
    result = openai_to_anthropic_response(openai_resp, model="test", use_reasoning=True)
    assert len(result["content"]) == 2
    assert "<thinking>" in result["content"][0]["text"]


def test_stream_conversion():
    """SSE 流式转换"""
    openai_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        'data: [DONE]\n',
    ]
    events = list(openai_stream_to_anthropic_stream(openai_lines, model="test", msg_id="msg_1"))
    # 解析所有事件
    parsed = []
    for e in events:
        lines = e.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        data = json.loads(lines[1].replace("data: ", ""))
        parsed.append((event_type, data))

    assert parsed[0][0] == "message_start"
    assert parsed[1][0] == "content_block_start"
    # 找 text_delta 事件
    deltas = [(t, d) for t, d in parsed if t == "content_block_delta"]
    assert deltas[0][1]["delta"]["text"] == "Hello"
    assert deltas[1][1]["delta"]["text"] == " world"
    assert parsed[-1][0] == "message_stop"


def test_stream_tool_calls():
    """流式 tool_calls 转换"""
    openai_lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\""}}]},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"hi\\"}"}}]},"finish_reason":null}]}\n',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        'data: [DONE]\n',
    ]
    events = list(openai_stream_to_anthropic_stream(openai_lines, model="test", msg_id="msg_2"))
    parsed = []
    for e in events:
        lines = e.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        data = json.loads(lines[1].replace("data: ", ""))
        parsed.append((event_type, data))

    # 应包含 tool_use content_block_start
    tool_starts = [(t, d) for t, d in parsed if t == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0][1]["content_block"]["name"] == "search"

    # message_delta 应包含 stop_reason=tool_use
    msg_delta = [(t, d) for t, d in parsed if t == "message_delta"]
    assert msg_delta[0][1]["delta"]["stop_reason"] == "tool_use"
