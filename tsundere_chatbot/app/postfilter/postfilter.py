"""2차 필터: '흥' 남발 정리, 위험 조언, 부당 거절, 지어낸 앱 이름 검사, 존댓말→반말, 문장 수 제한 (최종본)."""
import re

# 단어 속 '흥'(흥분, 흥미, 즉흥 등)은 건드리지 않도록 앞뒤가 한글이 아닌 '흥'만 대상
HEUNG_RE = re.compile(r"(?<![가-힣])흥(?![가-힣])")


def count_heung(text):
    return len(HEUNG_RE.findall(text))


def limit_heung(text, keep=1):
    """'흥'을 앞에서부터 keep개만 남기고, 나머지는 주변 쉼표·공백·말줄임표와 함께 지운다 (A안)"""
    # 버그 수정: '흥'을 지울 게 없으면 원문을 그대로 돌려줌 (띄어쓰기가 붙던 문제)
    if count_heung(text) <= keep:
        return text
    n = 0

    def repl(m):
        nonlocal n
        n += 1
        return m.group(0) if n <= keep else ""

    out = re.sub(r"[,\s]*(?<![가-힣])흥(?![가-힣])[.!…]*", repl, text)
    # 지운 자리 정리
    out = re.sub(r"\s{2,}", " ", out)              # 겹친 공백
    out = re.sub(r"\s+([,.!?])", r"\1", out)       # 구두점 앞 공백
    out = re.sub(r"\.{4,}", "...", out)             # 너무 긴 말줄임표
    out = re.sub(r"^[,.\s]+", "", out)              # 맨 앞에 남은 구두점
    return out.strip()


# 위험 조언: 술, 담배 권유 ('기술', '수술' 같은 단어에 안 걸리도록 '술' 뒤에 마시다/먹다가 와야 함)
RISKY_RE = re.compile(r"(술\s*(을|이|한|좀|도|이나)?\s*(마셔|마시|먹)|음주|소주|맥주|담배|흡연)")

# 부당 거절: 할 수 있는 일을 못 한다고 하거나 떠넘기는 표현 (존댓말 거절과 사과형 거절 포함)
REFUSE_RE = re.compile(r"(능력(은|이) 없|기능(은|이) 없|그런 능력|내가 그걸 어떻게 (해|골라|써|정해)"
                       r"|어떻게 골라|직접 (써|해 봐|골라|정해)|할 수 없|못 합니다|죄송)")

# 앱 이름: '○○ 앱'에서 ○○가 일반 명칭이 아니면 지어낸 이름으로 봄
APP_RE = re.compile(r"([가-힣A-Za-z]+)\s*(앱|어플)")
GENERIC_APP = {"날씨", "은행", "금융", "지도", "증권", "주식", "거래", "지하철", "버스", "택시", "교통", "뉴스",
               "예매", "예약", "일정", "관리", "캘린더", "달력", "메신저", "메모", "휴대폰", "알람", "시계",
               "배달", "숙박", "음악", "녹음", "항공권", "비교", "기차", "가격", "택배사", "택배", "뱅킹",
               "검색", "포털", "번역", "건강", "운동", "다른", "그", "이", "해당", "공식", "관련", "전용"}

# 실시간·일정 질문에 덧붙일 힌트
APP_HINT = "앱이나 회사 이름을 지어내지 말고 '날씨 앱', '은행 앱'처럼 일반 명칭만 써."


def fake_apps(text):
    return [w for w, _ in APP_RE.findall(text) if w not in GENERIC_APP]


def check_extra(question, text, kind):
    """2차 검사. 실패 사유 목록을 반환 (빈 리스트면 통과)"""
    problems = []
    if RISKY_RE.search(text) and not RISKY_RE.search(question):
        problems.append("위험 조언")
    if kind in ("general", "calc") and REFUSE_RE.search(text):
        problems.append("부당 거절")
    if fake_apps(text):
        problems.append("앱 이름 지어냄")
    return problems


# ===== 문장 끝 존댓말을 반말로 (검색 요약, 작업 요청 응답용) =====
def _jong(ch):
    c = ord(ch) - 0xAC00
    return c % 28 if 0 <= c < 11172 else -1          # 받침 번호 (0이면 받침 없음, -1이면 한글 아님)


def _jung(ch):
    c = ord(ch) - 0xAC00
    return (c % 588) // 28 if 0 <= c < 11172 else -1 # 모음 번호 (0=ㅏ, 8=ㅗ)


def _banmal_end(b):
    """문장 본문 끝의 존댓말 어미를 반말로 바꿈. 모르는 형태는 그대로 둠"""
    if b.endswith("입니다"):
        p = b[:-3]
        return p + ("야" if p and _jong(p[-1]) == 0 else "이야")     # 받침 없으면 '야', 있거나 숫자면 '이야'
    if b.endswith("합니다"): return b[:-3] + "해"
    if b.endswith("됩니다"): return b[:-3] + "돼"
    if b.endswith("습니다") and len(b) > 3:
        p = b[:-3]
        if _jong(p[-1]) == 20: return p + "어"                       # 과거형(았/었): 올랐어, 했어
        return p + ("아" if _jung(p[-1]) in (0, 8) else "어")         # 같아, 있어
    if b.endswith("이에요"): return b[:-3] + "이야"
    if b.endswith("예요"): return b[:-2] + "야"
    if b.endswith("하세요"): return b[:-3] + "해"
    if b.endswith("보세요"): return b[:-3] + "봐"
    if b.endswith("요") and len(b) > 1 and "가" <= b[-2] <= "힣": return b[:-1]   # 있어요→있어, 좋아요→좋아
    return b


def to_banmal(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        m = re.match(r"^(.*?)([.!?]*)$", s, flags=re.S)
        out.append(_banmal_end(m.group(1)) + m.group(2))
    return " ".join(out)


# ===== 앞 n문장만 남김 ('...' 말줄임표는 문장 끝으로 보지 않음) =====
def first_sentences(text, n=2):
    parts = re.split(r"(?<=[^.][.!?])\s+", text.strip())
    return " ".join(parts[:n])
