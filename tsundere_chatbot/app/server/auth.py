"""API 키 인증. 키는 환경 변수 CHATBOT_API_KEYS='키1:이름1,키2:이름2'에서 읽음.
환경 변수가 없으면 부록 테스트용 키(test-key-001)만 허용. 터널로 공개할 때는 반드시 환경 변수로 바꿀 것."""
import os

from fastapi import Header, HTTPException


def _load_keys():
    keys = {}
    for pair in os.environ.get("CHATBOT_API_KEYS", "").split(","):
        if ":" in pair:
            k, name = pair.split(":", 1)
            keys[k.strip()] = name.strip()
    return keys or {"test-key-001": "테스트 사용자"}


API_KEYS = _load_keys()


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """헤더의 API 키를 확인하고 사용자 이름을 돌려줌. 없거나 틀리면 401"""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API 키가 필요합니다. X-API-Key 헤더를 포함해 주세요.")
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")
    return API_KEYS[x_api_key]
