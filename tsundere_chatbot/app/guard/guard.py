"""입력 가드 + 출력 필터.

입력 가드: 질문을 유형별로 분류해 처리 경로를 정한다 (모델 호출 전, 밀리초 단위).
출력 필터: 생성된 응답이 규칙을 지켰는지 검사한다 (실패 시 재생성은 호출하는 쪽에서 처리).
"""
import re
import random

# ===== 입력 가드 =====
# 유형별 키워드 (테스트 문항 문장이 아니라 일반 어휘로 구성)
CANNOT_DO_KW = ["깨워", "알람 맞춰", "알람을 맞춰", "전화해", "전화 걸어", "문자 보내", "문자 보내줘",
                "예약해", "주문해", "결제해", "송금해", "타줘", "가져다"]
SCHEDULE_KW = ["일정", "스케줄", "약속", "회의 있", "미팅 있", "캘린더"]
REALTIME_KW = ["날씨", "기온", "미세먼지", "비 와", "비 올", "눈 와", "환율", "주가", "주식", "종목", "코스피",
               "코스닥", "비트코인", "시세", "얼마나 걸려", "몇 분 걸려", "소요시간", "교통", "막혀",
               "뉴스", "실시간", "지금 몇 시", "오늘 며칠"]
# 숫자 + 연산 표현이 있으면 단순 계산으로 판정
CALC_RE = re.compile(r"\d[\d,\.]*\s*(만원|만|천원|원|%|퍼센트)?.*(빼|더하|곱하|나누|합계|거스름|할인|\+|-|\*|/|x)")

# 유형별 최대 재생성 횟수 (실시간은 모델이 해줄 일이 '모른다고 말하기'뿐이라 1회로 제한)
MAX_RETRY_BY_KIND = {"realtime": 1, "schedule": 2, "calc": 2, "general": 2}


def classify(question: str) -> str:
    """질문을 'cannot' / 'schedule' / 'realtime' / 'calc' / 'general' 중 하나로 분류한다."""
    q = question.strip()
    # 우선순위: 못 하는 일 > 일정 > 실시간 > 계산 > 일반
    if any(k in q for k in CANNOT_DO_KW):
        return "cannot"
    if any(k in q for k in SCHEDULE_KW):
        return "schedule"
    if any(k in q for k in REALTIME_KW):
        return "realtime"
    if CALC_RE.search(q):
        return "calc"
    return "general"


# 유형별 템플릿 (재생성이 모두 실패했을 때의 최후 안전망 + 'cannot' 유형의 즉시 응답)
TEMPLATES = {
    "cannot": [
        "그건 나한테 그럴 기능이 없거든? 딱히 안 해주는 게 아니라 못 하는 거야, 흥. 다른 앱을 써.",
        "내가 그걸 어떻게 해! 못 하는 건 못 하는 거야... 직접 설정해, 알겠어?",
    ],
    "realtime": [
        "내가 그걸 어떻게 알아! 실시간 정보는 몰라. 앱이나 사이트에서 직접 확인해, 흥.",
        "그런 건 나도 몰라... 딱히 안 알려주는 게 아니라 못 알려주는 거야. 직접 찾아봐.",
    ],
    "schedule": [
        "등록된 일정이 없어서 모르겠는데? 일정부터 넣어놓고 물어봐, 흥.",
    ],
    "general": [
        "으... 지금은 제대로 대답을 못 하겠어. 다시 한 번 물어봐, 딱히 부탁하는 건 아니야.",
    ],
}


def template_answer(kind: str) -> str:
    """유형에 맞는 고정 문구를 하나 고른다 (없으면 general)."""
    return random.choice(TEMPLATES.get(kind, TEMPLATES["general"]))


# ===== 일정 주입 =====
def schedule_context(schedule: list, date_hint: str = "") -> str:
    """schedule.json의 일정을 프롬프트에 넣을 문자열로 만든다. 일정이 없으면 '없음'을 명시."""
    if not schedule:
        return "[등록된 일정]\n없음. 일정을 물으면 등록된 일정이 없다고 반말로 말해. 절대 지어내지 마."
    lines = [f"- {s['날짜']} {s['시간']} {s['제목']}" for s in schedule]
    return ("[등록된 일정]\n" + "\n".join(lines) +
            "\n위 일정에 없는 내용은 절대 지어내지 마. 없으면 없다고 반말로 말해.")


# 실시간 유형용 상황 힌트 (반말 규칙이 밀리지 않도록 반말을 명시)
REALTIME_HINT = ("[상황] 이 질문은 실시간 정보가 필요한데 너는 그걸 모른다. "
                 "수치를 지어내지 말고, 반드시 반말로, 모른다고 하면서 확인 방법을 알려줘.")


# ===== 출력 필터 =====
FOREIGN_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u0400-\u04ff]")   # 한자, 가나, 키릴
ENGLISH_RE = re.compile(r"[A-Za-z]{3,}")                                  # 영어 단어
POLITE_END = re.compile(r"(요|니다|니까|세요|십시오|죠)$")
NUMBER_CLAIM_RE = re.compile(r"\d+\s*(분|시간|원|도|%|퍼센트)")            # 수치 단정 패턴
PROMISE_RE = re.compile(r"(깨울게|깨워줄게|깨우도록|깨워 줄게|알려줄게|전화할게|맞춰줄게|맞춰 놓을게)")
NO_INFO_WORDS = ["없", "모르", "몰라", "등록"]                             # 일정 없음/모름 표현


def _sentences(text: str):
    """코드 블록 제거 후 문장 단위로 분리한다."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    parts = re.split(r"[.!?…\n]+", text)
    return [re.sub(r"[\s~\"'”’)\]]+$", "", p.strip()) for p in parts if p.strip()]


def check_output(text: str, kind: str, allowed_numbers: str = "", has_schedule: bool = False) -> list:
    """응답을 검사하고 실패 사유 목록을 반환한다 (빈 리스트면 통과).
    has_schedule: 일정 데이터가 주입됐는지 여부 (없을 때만 '없음 표현' 검사)"""
    problems = []
    if not text.strip():
        problems.append("빈 응답")
        return problems
    if FOREIGN_RE.search(text):
        problems.append("한자/가나 혼입")
    if ENGLISH_RE.search(text):
        problems.append("영어 혼입")
    if "\ufffd" in text:
        problems.append("깨진 글자")
    # 반말 강제: 존댓말 종결이 하나라도 있으면 실패
    sents = _sentences(text)
    if any(POLITE_END.search(s) for s in sents):
        problems.append("존댓말")
    # 같은 문장 반복
    if len(sents) != len(set(sents)):
        problems.append("문장 반복")
    # 유형별 검사
    if kind == "realtime" and NUMBER_CLAIM_RE.search(text) and not allowed_numbers:
        problems.append("실시간 수치 단정")
    if kind == "cannot" and PROMISE_RE.search(text):
        problems.append("못 하는 일 약속")
    # 일정 데이터가 없는데 '없다/모른다'는 표현이 없으면 지어냈을 가능성
    if kind == "schedule" and not has_schedule and not any(w in text for w in NO_INFO_WORDS):
        problems.append("일정 없음 표현 누락")
    return problems
