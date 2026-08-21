"""
网络策略工具：直连优先(3s 探测)，失败/超时则走本机代理。
"""

import os
import socket


PROXY = "http://127.0.0.1:7897"


def _configure_network():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        print("[网络] 直连 huggingface.co 可用")
    except OSError:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.setdefault(var, PROXY)
        print(f"[网络] 直连失败，走代理 {PROXY}")
