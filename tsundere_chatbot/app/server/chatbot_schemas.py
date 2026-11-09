"""챗봇 API 스키마 (부록 3.2절 구조 + 응답에 경로와 지연 시간 추가)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "bot", "assistant"] = Field(..., description="'user' 또는 'bot'('assistant'도 허용)")
    content: str = Field(..., min_length=1, max_length=500, description="메시지 내용 (500자 이하)")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=40,
                                    description="대화 기록. 마지막 메시지가 사용자의 현재 입력")
    max_new_tokens: int = Field(default=150, ge=10, le=300)
    temperature: float = Field(default=0.5, gt=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def last_is_user(cls, v):
        if v[-1].role != "user":
            raise ValueError("마지막 메시지는 사용자 입력이어야 합니다")
        return v

    model_config = {"json_schema_extra": {"examples": [
        {"messages": [{"role": "user", "content": "내일 일정 알려줘"}], "max_new_tokens": 150, "temperature": 0.5}]}}


class ChatResponse(BaseModel):
    success: bool
    response: str = Field(description="생성된 응답")
    model_name: str
    user: Optional[str] = None
    route: str = Field(description="처리 경로: model/base/schedule/search/python/template/rule/fallback")
    kind: str = Field(description="가드가 분류한 질문 유형")
    latency_sec: float


class ScheduleItem(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-11-07"])
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$", examples=["11:00"])
    title: str = Field(..., min_length=1, max_length=50, examples=["도서관 가기"])
