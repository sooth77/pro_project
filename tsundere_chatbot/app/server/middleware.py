"""요청 로깅 미들웨어 (순수 ASGI): 경로, 상태 코드, 처리 시간을 기록."""
import logging
import time

logger = logging.getLogger("ml_api")


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0, info = time.perf_counter(), {"status": 0}

        async def send_and_watch(message):
            if message["type"] == "http.response.start":
                info["status"] = message["status"]
                message["headers"].append((b"x-process-time", f"{time.perf_counter() - t0:.3f}".encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_and_watch)
        finally:
            line = f"{scope['method']} {scope['path']} -> {info['status']} ({time.perf_counter() - t0:.2f}초)"
            (logger.warning if info["status"] >= 400 else logger.info)(line)
