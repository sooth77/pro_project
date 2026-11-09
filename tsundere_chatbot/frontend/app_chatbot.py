"""츤데레 비서 채팅 화면 (부록 4.1절 구조 + 일정 보기·추가 + 응답 경로 표시)."""
import datetime as dt
import os

import requests
import streamlit as st

API_BASE = os.environ.get("CHATBOT_API_BASE", "http://localhost:8000")
st.set_page_config(page_title="츤데레 비서", page_icon="💢", layout="centered")


def call(method, path, api_key, **kw):
    """API 호출 공통 처리: 연결 실패, 401, 그 외 오류를 화면에 표시"""
    try:
        r = requests.request(method, f"{API_BASE}{path}", headers={"X-API-Key": api_key}, timeout=60, **kw)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("서버에 연결할 수 없습니다.")
    except requests.exceptions.HTTPError as e:
        st.error("인증 실패. API 키를 확인하세요." if e.response.status_code == 401
                 else f"서버 오류 (HTTP {e.response.status_code})")
    except Exception as e:
        st.error(f"오류: {type(e).__name__}")
    return None


# ===== 사이드바: 설정, 서버 상태, 일정 =====
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API 키", type="password")          # 공개 URL을 고려해 기본값을 채우지 않음
    temperature = st.slider("Temperature", 0.1, 1.5, 0.5, step=0.1)
    max_tokens = st.slider("최대 생성 토큰", 30, 300, 150, step=10)

    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success("서버 연결됨") if h.get("status") == "healthy" else st.warning("모델 로딩 중...")
        st.caption(f"모델: {h.get('model')}")
    except Exception:
        st.error("서버 연결 실패")

    if st.button("대화 초기화"):
        st.session_state["chat"] = []
        st.rerun()

    st.divider()
    st.subheader("일정")
    if api_key:
        data = call("GET", "/schedule", api_key)
        for x in (data or {}).get("items", []):
            st.caption(f"{x['date']} {x['time']}  {x['title']}")
        with st.form("add_schedule", clear_on_submit=True):
            d = st.date_input("날짜", value=dt.date.today())
            t = st.time_input("시간", value=dt.time(10, 0))
            title = st.text_input("제목")
            if st.form_submit_button("일정 추가") and title.strip():
                call("POST", "/schedule", api_key,
                     json={"date": d.isoformat(), "time": t.strftime("%H:%M"), "title": title.strip()})
                st.rerun()
    else:
        st.caption("API 키를 입력하면 일정을 볼 수 있어요.")

# ===== 본문: 채팅 =====
st.title("💢 츤데레 비서")
st.caption("딱히 너를 도와주려는 건 아니거든. 일정, 날씨, 환율, 계산도 물어봐.")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("route"):
            st.caption(f"경로: {m['route']} · {m['latency']}초")

q = st.chat_input("메시지를 입력하세요...")
if q:
    if not api_key:
        st.warning("사이드바에 API 키를 먼저 입력하세요.")
        st.stop()
    st.session_state["chat"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("생각 중... 딱히 기다리게 하려는 건 아니야"):
            msgs = [{"role": "user" if m["role"] == "user" else "bot", "content": m["content"]}
                    for m in st.session_state["chat"]][-20:]               # 최근 20개만 전송
            res = call("POST", "/chat", api_key,
                       json={"messages": msgs, "max_new_tokens": max_tokens, "temperature": temperature})
        if res and res.get("success"):
            st.write(res["response"])
            st.caption(f"경로: {res['route']} · {res['latency_sec']}초")
            st.session_state["chat"].append({"role": "assistant", "content": res["response"],
                                             "route": res["route"], "latency": res["latency_sec"]})
        else:
            st.session_state["chat"].pop()        # 실패한 질문은 기록에서 뺌
