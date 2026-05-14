"""
HTTP 代理服务器

接收 Anthropic /v1/messages 请求，转换为 OpenAI /v1/chat/completions 请求，
转发给后端，并将响应转换回 Anthropic 格式返回。
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import threading
from typing import Callable, Dict, Optional

from anthropic_openai_adapter.converter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    openai_stream_to_anthropic_stream,
)

logger = logging.getLogger("anthropic_openai_adapter")


class AdapterHandler(BaseHTTPRequestHandler):
    """处理 Anthropic → OpenAI 的请求转换"""

    # 由 server 实例注入
    backend_url: str = ""
    api_key: str = ""
    model_map: Optional[Dict[str, str]] = None
    use_reasoning: bool = False
    on_request: Optional[Callable] = None

    def do_POST(self):
        if "/v1/messages" not in self.path:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        original_model = body.get("model", "")
        is_stream = body.get("stream", False)
        openai_payload = anthropic_to_openai_request(body, model_map=self.model_map)

        if is_stream:
            openai_payload["stream"] = True

        # 可选钩子：允许外部修改请求
        if self.on_request:
            openai_payload = self.on_request(openai_payload) or openai_payload

        url = f"{self.backend_url}/v1/chat/completions"
        req_body = json.dumps(openai_payload).encode()
        req = Request(url, data=req_body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        # 透传客户端 headers（可选）
        for header in ("X-Request-Id", "X-Trace-Id"):
            val = self.headers.get(header)
            if val:
                req.add_header(header, val)

        try:
            resp = urlopen(req, timeout=120)
        except HTTPError as e:
            error_body = e.read().decode(errors="replace")
            logger.error(f"Backend returned {e.code}: {error_body[:500]}")
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": error_body[:500]},
            }).encode())
            return
        except Exception as e:
            logger.error(f"Backend request failed: {e}")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            }).encode())
            return

        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            lines = []
            for raw_line in resp:
                lines.append(raw_line.decode(errors="replace"))

            msg_id = f"msg_{id(self)}"
            for sse_chunk in openai_stream_to_anthropic_stream(
                lines, original_model, msg_id, use_reasoning=self.use_reasoning
            ):
                self.wfile.write(sse_chunk.encode())
                self.wfile.flush()
        else:
            resp_body = resp.read()
            try:
                openai_resp = json.loads(resp_body)
            except json.JSONDecodeError:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b"Invalid JSON from backend")
                return

            anthropic_resp = openai_to_anthropic_response(
                openai_resp, model=original_model, use_reasoning=self.use_reasoning
            )

            result = json.dumps(anthropic_resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

    def do_GET(self):
        """健康检查"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        logger.debug(f"adapter: {format % args}")


class ThreadingAdapterServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务"""
    daemon_threads = True


def create_server(
    backend_url: str,
    api_key: str = "",
    host: str = "127.0.0.1",
    port: int = 8081,
    model_map: Optional[Dict[str, str]] = None,
    use_reasoning: bool = False,
    on_request: Optional[Callable] = None,
) -> ThreadingAdapterServer:
    """
    创建 adapter 服务器实例（不启动）。

    Args:
        backend_url: OpenAI 兼容后端地址（如 http://api.openai.com 或本地 vLLM）
        api_key: 后端 API Key
        host: 监听地址
        port: 监听端口
        model_map: 模型名映射表 {"anthropic_name": "backend_name"}
        use_reasoning: 是否透传 reasoning 字段
        on_request: 可选请求钩子 (openai_payload) -> openai_payload

    Returns:
        ThreadingAdapterServer 实例
    """
    # 通过类属性注入配置
    AdapterHandler.backend_url = backend_url.rstrip("/")
    AdapterHandler.api_key = api_key
    AdapterHandler.model_map = model_map
    AdapterHandler.use_reasoning = use_reasoning
    AdapterHandler.on_request = on_request

    server = ThreadingAdapterServer((host, port), AdapterHandler)
    return server


def start_server(
    backend_url: str,
    api_key: str = "",
    host: str = "127.0.0.1",
    port: int = 8081,
    model_map: Optional[Dict[str, str]] = None,
    use_reasoning: bool = False,
    on_request: Optional[Callable] = None,
    daemon: bool = True,
) -> ThreadingAdapterServer:
    """
    创建并在后台线程启动 adapter 服务器。

    Args:
        daemon: 是否为 daemon 线程（主进程退出时自动终止）
        其他参数同 create_server

    Returns:
        ThreadingAdapterServer 实例
    """
    server = create_server(
        backend_url=backend_url,
        api_key=api_key,
        host=host,
        port=port,
        model_map=model_map,
        use_reasoning=use_reasoning,
        on_request=on_request,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Anthropic→OpenAI adapter listening on {host}:{port} → {backend_url}")
    return server
