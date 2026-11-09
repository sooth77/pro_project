"""츤데레 비서 FastAPI 서버 (부록 3.3절 구조, lifespan으로 모델 로드)."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.routes import schedule as sch
from app.server.auth import verify_api_key
from app.server.chatbot_schemas import ChatRequest, ChatResponse, ScheduleItem
from app.server.error_handlers import register_error_handlers
from app.server.logger_config import setup_logger
from app.server.middleware import RequestLoggingMiddleware

logger = setup_logger("chatbot_api")
setup_logger("ml_api")                       # 에러 핸들러와 미들웨어가 쓰는 로거
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chatbot")
STATE = {"bot": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.model.chatbot_model import TsundereBot
    logger.info("모델 로드 시작")
    STATE["bot"] = await asyncio.get_running_loop().run_in_executor(executor, TsundereBot)
    if not os.path.exists(sch.PATH):
        sch.seed_demo()                      # 일정 파일이 없으면 시연용 일정 생성
    logger.info(f"모델 로드 완료: {STATE['bot'].model_name}")
    yield
    STATE["bot"] = None


app = FastAPI(title="Tsundere Assistant API",
              description="츤데레 AI 비서 (Qwen2.5-1.5B + LoRA, 일정·검색·계산 포함, 인증 필요)",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)


@app.get("/health", tags=["System"])
async def health():
    bot = STATE["bot"]
    return {"status": "healthy" if bot else "loading", "model": bot.model_name if bot else None}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest, user: str = Depends(verify_api_key)):
    bot = STATE["bot"]
    if bot is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    msgs = [m.model_dump() for m in req.messages]
    try:
        r = await asyncio.get_running_loop().run_in_executor(
            executor, bot.respond, msgs, req.max_new_tokens, req.temperature)
    except Exception:
        logger.exception("응답 생성 실패")                            # 상세 내용은 로그에만
        raise HTTPException(status_code=500, detail="응답 생성 중 오류가 발생했습니다.")
    logger.info(f"채팅 - 사용자: {user}, 메시지 {len(msgs)}개, 경로 {r['route']}, {r['latency_sec']}초")
    return ChatResponse(success=True, response=r["text"], model_name=bot.model_name, user=user,
                        route=r["route"], kind=r["kind"], latency_sec=r["latency_sec"])


@app.get("/schedule", tags=["Schedule"])
async def get_schedule(user: str = Depends(verify_api_key)):
    return {"success": True, "items": sch.load()}


@app.post("/schedule", tags=["Schedule"])
async def add_schedule(item: ScheduleItem, user: str = Depends(verify_api_key)):
    sch.add(item.date, item.time, item.title)
    return {"success": True, "items": sch.load()}
