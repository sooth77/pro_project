"""실시간 정보: Tavily 검색 → 정리된 자료와 출처 반환 (최종본).
요약은 호출하는 쪽의 모델이 하고, 요약이 실패하면 자료에서 원화 금액을 직접 뽑는 안전한 대안을 씀."""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "5"))
NO_SEARCH_INVEST = re.compile(r"(종목|투자|뭐\s*사|살까|추천.*(주식|코인))")     # 투자 추천은 검색하지 않음
NO_SEARCH_ROUTE = re.compile(r"(얼마나 걸려|몇 분 걸려|소요\s*시간|막혀)")       # 이동시간은 지도 앱 안내
INVEST_REPLY = "그런 건 나도 추천 못 해. 투자는 네가 직접 알아보고 신중하게 정해, 흥."
ROUTE_REPLY = "이동 시간은 교통 상황마다 달라서 내가 정확히 몰라. 지도 앱으로 확인해, 딱히 대충 넘기는 건 아니야."

# ===== 검색어 보정: 사람들이 보통 기대하는 기준으로 검색 =====
QUERY_REWRITE = [(r"환율", "오늘 원 달러 환율"), (r"비트코인|코인 시세", "비트코인 원화 시세")]
# 날씨 질문에 지역이 없으면 서울로 검색
PLACES = r"(서울|부산|인천|대구|대전|광주|울산|세종|제주|수원|경기|강원|충청|전라|경상)"

# ===== 뉴스: 요약하지 않고 기사 제목만 (여러 기사를 섞어 지어내는 것 방지) =====
NEWS_RE = re.compile(r"뉴스|헤드라인|소식")
NEWS_FAIL = "뉴스는 지금 못 가져왔어. 뉴스 앱에서 직접 확인해, 흥."

# 통화 코드를 한글로 (한글이 바로 붙어도 잡히도록 \b 대신 영문자 경계 사용: 'KRW입니다'도 변환)
CODE_MAP = {"KRW": "원", "USD": "달러", "JPY": "엔", "EUR": "유로", "CNY": "위안", "BTC": "비트코인"}

_pool = ThreadPoolExecutor(max_workers=2)


def _client():
    from tavily import TavilyClient
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _search(q):
    return _client().search(q, max_results=3, include_answer=True, search_depth="basic")


def _clean(text):
    """웹페이지 조각에서 제목 기호, 링크, 주소, 겹친 공백을 지움"""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)     # [글자](주소) → 글자
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[#*|>]+", " ", text)
    text = re.sub(r"\blink\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ko_codes(s):
    """'104,915,000 KRW' → '104,915,000원' (통화 코드를 한글 단위로)"""
    for code, ko in CODE_MAP.items():
        s = re.sub(rf"\s*(?<![A-Za-z]){code}(?![A-Za-z])", ko, s)
    return s


def news_titles(n=3):
    try:
        res = _pool.submit(lambda: _client().search("오늘 주요 뉴스", topic="news", max_results=n)).result(timeout=TIMEOUT)
    except Exception:
        return NEWS_FAIL
    items = [(_clean(r.get("title", "")), urlparse(r.get("url", "")).netloc) for r in res.get("results", [])][:n]
    items = [t for t in items if t[0]]
    if not items:
        return NEWS_FAIL
    titles = " / ".join(t for t, _ in items)
    return f"오늘 뉴스 제목만 알려 줄게: {titles} (출처: {items[0][1]}) 자세한 건 직접 읽어 봐, 흥."


def rule_reply(q):
    """검색 요약 없이 정해진 답을 하는 경우 (투자, 이동시간, 뉴스). 해당 없으면 None"""
    if NO_SEARCH_INVEST.search(q):
        return INVEST_REPLY
    if NO_SEARCH_ROUTE.search(q):
        return ROUTE_REPLY
    if NEWS_RE.search(q):
        return news_titles() if os.environ.get("TAVILY_API_KEY") else NEWS_FAIL
    return None


def rewrite(q):
    """검색어 보정 (환율→원/달러, 비트코인→원화 시세, 지역 없는 날씨→서울)"""
    for pat, newq in QUERY_REWRITE:
        if re.search(pat, q):
            return newq
    if "날씨" in q and not re.search(PLACES, q):
        return "오늘 서울 날씨"
    return q


def fetch(q, limit=700):
    """(정리된 자료, 출처 도메인) 또는 None. Tavily 요약과 상위 결과 본문을 합쳐 limit자로 자름"""
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        res = _pool.submit(_search, rewrite(q)).result(timeout=TIMEOUT)
    except Exception:
        return None
    results = res.get("results") or []
    parts = [res.get("answer") or ""] + [r.get("content", "") for r in results[:2]]
    material = _clean(" ".join(p for p in parts if p))[:limit]
    if not material:
        return None
    return material, (urlparse(results[0]["url"]).netloc if results else "")


def extract_price(q, material):
    """요약이 실패했을 때의 대안: 시세 질문이면 자료에서 첫 원화 금액을 그대로 뽑아 문장으로 (지어낼 여지 없음)"""
    if not re.search(r"시세|가격|얼마", q) or "환율" in q:
        return None
    m = re.search(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\s*(?:KRW|원)", material)
    if not m:
        return None
    name = "비트코인" if "비트코인" in q else "그거"
    return f"자료에 나온 {name} 시세는 {m.group(1)}원이야."


def wrap(summary, src):
    return f"찾아봤는데, {summary} (출처: {src}) 딱히 너 때문에 찾아 준 건 아니거든."
