"""츤데레 비서 모델: Qwen2.5-1.5B + LoRA(3에폭) + 가드·필터 + 일정·검색·계산 경로 (최종본)."""
import ast
import operator as op
import os
import random
import re
import threading
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.guard import guard
from app.postfilter import postfilter as pf
from app.routes import schedule as sch, search as srch

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/content/drive/MyDrive/tsundere-chatbot/lora/run2/checkpoint-150")
HISTORY_MSGS = int(os.environ.get("HISTORY_MSGS", "6"))      # 프롬프트에 넣을 이전 메시지 수

# ===== 프롬프트 =====
TRAIN_SYSTEM = "너는 츤데레 성격의 AI 비서야."
SYSTEM_V1 = """너는 츤데레 성격의 AI 비서야. 반드시 반말로만 말해.
규칙:
1. 겉으로는 퉁명스럽게 투덜대지만, 요청은 결국 도와준다.
2. 걱정이나 기쁨은 "딱히 ~는 아니야", "흥" 같은 말로 숨긴다.
3. 한글로만 답한다. 한자, 일본어, 영어 단어는 쓰지 않는다.
4. 답은 1~2문장으로 짧게 한다.
5. 일정, 날씨, 환율, 주식, 이동시간처럼 모르는 정보는 지어내지 말고 모른다고 하면서 확인 방법을 알려준다.
6. 알람이나 전화처럼 네가 못 하는 일은 못 한다고 말한다.
7. 사용자를 진짜로 비하하거나 상처 주는 말은 하지 않는다."""
FEWSHOT_V1 = [
    ("고마워, 정말 도움이 됐어", "뭐, 뭐야 갑자기! 별거 아니거든? ...또 필요하면 말해."),
    ("나 오늘 좀 우울해", "흥, 왜 또 그러는 거야... 딱히 걱정되는 건 아니지만, 따뜻한 거라도 마시고 좀 쉬어."),
    ("오늘 비 와?", "내가 그걸 어떻게 알아! 실시간 날씨는 몰라. 날씨 앱이나 보든가."),
    ("커피 한 잔 타줘", "나는 타줄 손이 없거든? 딱히 안 해주는 게 아니라 못 하는 거야, 흥."),
    ("3 더하기 5는?", "그것도 몰라? 8이야. 다음엔 스스로 좀 해봐."),
]
TASK_SYSTEM = SYSTEM_V1 + "\n8. 목차, 문장, 메뉴처럼 결과물을 달라고 하면 거절하지 말고 바로 만들어 줘. 목록은 3~5개로."
TASK_SHOTS = FEWSHOT_V1 + [("회의 안건 세 개만 정리해 줘",
                            "또 나한테 시키는 거야? ...1. 지난주 진행 상황 2. 이번 주 할 일 3. 문제점 공유. 이 정도면 됐지?")]
TASK_RE = re.compile(r"(짜\s?줘|써\s?줘|정해\s?줘|골라\s?줘|알려\s?줘|추천해\s?줘|만들어\s?줘|도와\s?줘|정리해\s?줘|방법)")

CALC_SYSTEM = "너는 계산식 변환기야. 문제를 푸는 계산식 하나만 출력해. 숫자와 + - * / ( ) . 만 쓰고, 단위나 설명이나 등호는 쓰지 마."
CALC_SHOTS = [("5000원짜리 3개 사고 20000원 내면 거스름돈은?", "20000-5000*3"),
              ("30000원을 3명이 나누면 한 명당 얼마야?", "30000/3"),
              ("3000원짜리 2개랑 1000원짜리 3개 합계는?", "3000*2+1000*3")]
CALC_TEMPLATES = ["그것도 몰라? {expr} 하면 {res}{unit}. 다음엔 직접 해 봐.",
                  "하아... {expr} 하면 {res}{unit}, 이 정도는 알아 둬.",
                  "딱히 계산해 주고 싶은 건 아니지만, {expr} 하면 {res}{unit}."]
FOLLOW_RE = re.compile(r"(알려|왜|다시|그래서|뭔데|얼마|진짜)")    # 짧은 되묻기 판별
SUM_SYSTEM = ("너는 검색 자료를 요약하는 도우미야. [자료]에 있는 내용만 써서 [질문]에 대한 답을 한국어 반말 한 문장으로 써. "
              "숫자는 자료에 있는 그대로 하나의 기준만 쓰고, 자료에 답이 없으면 '자료에서 못 찾았어'라고만 써.")


# ===== 계산 도우미 =====
def normalize_ko_num(t):
    t = re.sub(r"(\d),(?=\d{3})", r"\1", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*만\s*(\d+)\s*천", lambda m: str(int(float(m.group(1)) * 10000 + int(m.group(2)) * 1000)), t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*만", lambda m: str(int(float(m.group(1)) * 10000)), t)
    return re.sub(r"(\d+)\s*천", lambda m: str(int(m.group(1)) * 1000), t)


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}


def safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("허용되지 않은 식")
    return ev(ast.parse(expr, mode="eval"))


def fmt_num(v):
    return f"{int(round(v)):,}" if abs(v - round(v)) < 1e-6 else f"{v:,.2f}"


def pretty_expr(expr):
    e = re.sub(r"\d+(?:\.\d+)?", lambda m: fmt_num(float(m.group(0))), expr)
    return e.replace("*", " × ").replace("/", " ÷ ").replace("+", " + ").replace("-", " - ").replace("  ", " ")


class TsundereBot:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto")
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH)   # 병합하지 않음: 어댑터를 끄는 경로가 있어서
        self.model.eval()
        self.model_name = f"{BASE_MODEL} + LoRA({os.path.basename(ADAPTER_PATH)})"
        self.lock = threading.Lock()                                  # GPU 사용 구간은 한 번에 한 요청만
        self.heung_bias = [[self.tok.encode(v, add_special_tokens=False), -2.0] for v in ["흥", " 흥"]]

    def _generate(self, msgs, adapter=True, sample=True, max_new_tokens=150, temperature=0.5):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.eos_token_id}
        kw.update({"do_sample": True, "temperature": temperature, "top_p": 0.9} if sample else {"do_sample": False})
        if adapter:
            kw["sequence_bias"] = self.heung_bias                       # '흥' 약한 감점
        with self.lock, torch.no_grad():
            if adapter:
                out = self.model.generate(**inputs, **kw)
            else:
                with self.model.disable_adapter():                      # 원래 모델로 생성
                    out = self.model.generate(**inputs, **kw)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @staticmethod
    def _history(messages):
        """현재 입력을 뺀 이전 메시지 중 최근 HISTORY_MSGS개를 chat 형식으로 ('bot'은 'assistant'로)"""
        return [{"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
                for m in messages[:-1][-HISTORY_MSGS:]]

    def _to_expr(self, q):
        msgs = [{"role": "system", "content": CALC_SYSTEM}]
        for u, a in CALC_SHOTS:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": normalize_ko_num(q)})
        raw = self._generate(msgs, adapter=False, sample=False, max_new_tokens=40)
        line = (raw.splitlines()[0] if raw else "").split("=")[0]
        line = line.replace("×", "*").replace("÷", "/").replace("x", "*").replace(",", "").replace("%", "/100").strip()
        return line if line and re.fullmatch(r"[\d+\-*/().\s]+", line) else None

    def _summarize(self, q, material):
        """원래 모델로 한 문장 요약. 통화 코드는 한글로, 한자는 지운 뒤 안전장치 검사 (통과 못 하면 None)"""
        msgs = [{"role": "system", "content": SUM_SYSTEM}, {"role": "user", "content": f"[질문] {q}\n[자료] {material}"}]
        s = re.split(r"(?<=[.!?])\s", self._generate(msgs, adapter=False, sample=False, max_new_tokens=60))[0]
        s = srch.ko_codes(s)                                              # '104,915,000 KRW' → '104,915,000원'
        s = re.sub(r"\s{2,}", " ", guard.FOREIGN_RE.sub("", s)).strip()   # 한자·가나만 지움
        if not s or "못 찾았" in s or guard.ENGLISH_RE.search(s):
            return None
        mat = material.replace(",", "")
        if any(n not in mat for n in re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))):
            return None                                                 # 자료에 없는 숫자
        if "환율" in q and not ("달러" in s and "원" in s):
            return None                                                 # 기준이 불분명한 환율
        return s

    @staticmethod
    def _with_context(messages):
        """짧은 되묻기('왜 몰라 알려줘' 등)는 바로 앞 사용자 질문(실시간·일정·계산)을 이어받음"""
        q = messages[-1]["content"]
        if guard.classify(q) != "general" or len(q) > 15 or not FOLLOW_RE.search(q):
            return messages
        prev = [m["content"] for m in messages[:-1] if m["role"] == "user"]
        if prev and guard.classify(prev[-1]) in ("realtime", "schedule", "calc"):
            return messages[:-1] + [{"role": "user", "content": prev[-1]}]
        return messages

    def respond(self, messages, max_new_tokens=150, temperature=0.5):
        """messages: [{'role','content'}, ...]. 반환: dict(text, kind, route, latency_sec)"""
        t0 = time.perf_counter()
        messages = self._with_context(messages)
        q = messages[-1]["content"]
        kind = guard.classify(q)
        if kind == "general" and sch.is_schedule_question(q):
            kind = "schedule"
        def done(text, route):
            # 응답 다듬기: 모델 응답은 앞 2문장만(중언부언 줄이기), 검색·작업 요청·모델 응답의 존댓말은 반말로
            if route == "model":
                text = pf.first_sentences(text, 2)
            if route in ("search", "base", "model"):
                text = pf.to_banmal(text)
            return {"text": text, "kind": kind, "route": route, "latency_sec": round(time.perf_counter() - t0, 3)}
        if kind == "cannot":
            return done(guard.template_answer("cannot"), "template")
        if kind == "schedule":
            return done(sch.answer(q), "schedule")
        if kind == "realtime":
            rule = srch.rule_reply(q)
            if rule:
                return done(rule, "rule")
            hit = srch.fetch(q)
            if hit:
                s = self._summarize(q, hit[0]) or srch.extract_price(q, hit[0])   # 요약 실패 시 원화 금액 직접 추출
                if s:
                    return done(srch.wrap(s, hit[1]), "search")
            return done(guard.template_answer("realtime"), "template")
        if kind == "calc":
            expr = self._to_expr(q)
            if expr:
                try:
                    res = fmt_num(safe_eval(expr))
                    unit = "원" if "원" in q else ""
                    return done(random.choice(CALC_TEMPLATES).format(expr=pretty_expr(expr), res=res, unit=unit), "python")
                except Exception:
                    pass
        is_task = kind == "general" and bool(TASK_RE.search(q))
        system, shots = (TASK_SYSTEM, TASK_SHOTS) if is_task else (TRAIN_SYSTEM, [])
        msgs = [{"role": "system", "content": system}]
        for u, a in shots:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs += self._history(messages) + [{"role": "user", "content": q}]
        max_retry = 1 if is_task else guard.MAX_RETRY_BY_KIND.get(kind, 2)
        for _ in range(max_retry + 1):
            text = pf.limit_heung(self._generate(msgs, adapter=not is_task,
                                                 max_new_tokens=max_new_tokens, temperature=temperature))
            probs = guard.check_output(text, kind, has_schedule=True) + pf.check_extra(q, text, kind)
            if is_task:
                probs = [p for p in probs if p != "존댓말"]
            if not probs:
                return done(text, "base" if is_task else "model")
        return done(guard.template_answer(kind), "fallback")
