"""일정: schedule.json 읽기/쓰기, 질문의 날짜 해석, 일정 답변 문장 만들기 (최종본)."""
import json
import os
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

PATH = os.environ.get("SCHEDULE_PATH", "data/schedule.json")
TZ = ZoneInfo("Asia/Seoul")
DAYS = "월화수목금토일"

# 날짜 표현 + 일정을 묻는 말이 같이 있으면 일정 질문으로 봄 ('토요일에 뭐 있어?' 등)
SCHED_HINT_RE = re.compile(r"(오늘|내일|모레|이번\s*주|다음\s*주|담주|[월화수목금토일]요일|\d{1,2}\s*월\s*\d{1,2}\s*일)"
                           r".*(뭐\s*있|있어|있나|있었|계획|약속|일정|스케줄|바빠)")


def today():
    return datetime.now(TZ).date()


def load():
    if not os.path.exists(PATH):
        return []
    return json.load(open(PATH, encoding="utf-8"))


def save(items):
    os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
    json.dump(sorted(items, key=lambda x: (x["date"], x["time"])), open(PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def add(d, t, title):
    """일정 추가 (화면의 입력 칸에서 호출할 함수). d='YYYY-MM-DD', t='HH:MM'"""
    items = load()
    items.append({"date": d, "time": t, "title": title})
    save(items)


def seed_demo(base=None):
    """시연용 가짜 일정을 오늘 기준으로 만듦 (실제 개인 일정은 넣지 않음)"""
    b = base or today()
    fri = b + timedelta(days=(4 - b.weekday()) % 7)                 # 이번 주 금요일
    next_tue = b + timedelta(days=7 - b.weekday() + 1)              # 다음 주 화요일
    save([
        {"date": b.isoformat(), "time": "19:00", "title": "헬스장"},
        {"date": (b + timedelta(days=1)).isoformat(), "time": "10:00", "title": "팀 회의"},
        {"date": (b + timedelta(days=1)).isoformat(), "time": "15:00", "title": "치과 예약"},
        {"date": fri.isoformat(), "time": "18:30", "title": "친구랑 저녁 약속"},
        {"date": next_tue.isoformat(), "time": "14:00", "title": "프로젝트 발표"},
    ])


def is_schedule_question(q):
    return bool(SCHED_HINT_RE.search(q))


def parse_range(q, base=None):
    """질문에서 날짜 범위를 찾아 (시작, 끝, 말할 이름)을 반환. 못 찾으면 오늘부터 7일"""
    b = base or today()
    week_start = b - timedelta(days=b.weekday())                    # 이번 주 월요일
    m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", q)
    if m:
        d = date(b.year, int(m.group(1)), int(m.group(2)))
        return d, d, f"{d.month}월 {d.day}일"
    m = re.search(r"(이번\s*주|다음\s*주|담주)?\s*([월화수목금토일])요일", q)
    if m:
        offset = 7 if m.group(1) and ("다음" in m.group(1) or "담" in m.group(1)) else 0
        d = week_start + timedelta(days=offset + DAYS.index(m.group(2)))
        if not m.group(1) and d < b:                                # '금요일'만 말했고 이미 지났으면 다음 주
            d += timedelta(days=7)
        return d, d, f"{d.month}월 {d.day}일 {m.group(2)}요일"
    for word, n in [("모레", 2), ("내일", 1), ("오늘", 0)]:
        if word in q:
            d = b + timedelta(days=n)
            return d, d, word
    if re.search(r"다음\s*주|담주", q):
        s = week_start + timedelta(days=7)
        return s, s + timedelta(days=6), "다음 주"
    if "이번 주" in q or "이번주" in q:
        return b, week_start + timedelta(days=6), "이번 주"
    return b, b + timedelta(days=6), "앞으로 일주일"


def _fmt_time(t):
    h, mi = map(int, t.split(":"))
    return f"{h}시" + (f" {mi}분" if mi else "")


def answer(q, base=None):
    """일정 질문에 대한 답 문장 (지어내지 않고 파일에 있는 것만 말함)"""
    s, e, label = parse_range(q, base)
    hits = [x for x in load() if s.isoformat() <= x["date"] <= e.isoformat()]
    if not hits:
        return f"{label}엔 등록된 일정 없어. 흥, 일정은 미리미리 넣어 둬."
    if s == e:
        items = ", ".join(f"{_fmt_time(x['time'])} {x['title']}" for x in hits)
    else:
        items = ", ".join(f"{int(x['date'][5:7])}월 {int(x['date'][8:])}일 {_fmt_time(x['time'])} {x['title']}" for x in hits)
    return f"{label}엔 {items} 있어. 딱히 챙겨 주는 건 아니거든, 잊지나 마."
