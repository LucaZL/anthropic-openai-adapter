"""
CLI 入口点

独立启动 adapter 作为反向代理：
  anthropic-openai-adapter --backend-url http://localhost:11434 --port 8081
"""

import argparse
import json
import logging
import signal
import sys

from anthropic_openai_adapter.server import create_server


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic Messages API → OpenAI Chat Completions reverse proxy"
    )
    parser.add_argument(
        "--backend-url",
        required=True,
        help="OpenAI-compatible backend URL (e.g. http://localhost:11434, https://api.openai.com)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the backend (also reads ADAPTER_API_KEY env var)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Listen host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Listen port (default: 8081)",
    )
    parser.add_argument(
        "--model-map",
        default=None,
        help='Model name mapping as JSON (e.g. \'{"claude-sonnet-4-5": "gpt-4o"}\')',
    )
    parser.add_argument(
        "--use-reasoning",
        action="store_true",
        help="Include reasoning field from backend response",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import os
    api_key = args.api_key or os.environ.get("ADAPTER_API_KEY", "")

    model_map = None
    if args.model_map:
        try:
            model_map = json.loads(args.model_map)
        except json.JSONDecodeError:
            print(f"Error: --model-map must be valid JSON", file=sys.stderr)
            sys.exit(1)

    server = create_server(
        backend_url=args.backend_url,
        api_key=api_key,
        host=args.host,
        port=args.port,
        model_map=model_map,
        use_reasoning=args.use_reasoning,
    )

    def _shutdown(sig, frame):
        print("\nShutting down...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"anthropic-openai-adapter v0.1.0")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Backend: {args.backend_url}")
    if model_map:
        print(f"Model map: {model_map}")
    print(f"Use reasoning: {args.use_reasoning}")
    print()

    server.serve_forever()


if __name__ == "__main__":
    main()
