# Streamlit 앱: 학습된 모델 목록에서 하나를 골라 손글씨 숫자를 분류
# 실행: mnist_app 폴더에서  streamlit run mnist_ui.py
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import datasets

from mnist_common import DATA_DIR, MODEL_DIR, SimpleClassifier, to_tensor

# ---------------- 페이지 설정과 테마 (반드시 첫 Streamlit 명령) ----------------
st.set_page_config(page_title="MNIST 손글씨 분류기", page_icon="✏️", layout="centered")

# config.toml이 적용되지 않아도 처음부터 연한 초록색으로 보이도록 CSS로 한 번 더 지정
st.markdown(
    """
    <style>
    .stApp { background-color: #EAF7EC; }                       /* 전체 배경: 연한 초록 */
    [data-testid="stHeader"] { background-color: #EAF7EC; }     /* 상단 바도 같은 색 */
    [data-testid="stExpander"] details { background-color: #D3EFD8; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- 함수 ----------------
def to_mnist_style(img):
    """흰 글씨/검은 배경 이미지를 MNIST처럼 글씨만 잘라 20x20으로 줄이고 28x28 가운데에 놓는다"""
    arr = np.array(img)
    ys, xs = np.nonzero(arr > 30)
    if len(xs) == 0:
        return None                                   # 아무것도 안 그림
    digit = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = 20 / max(digit.size)
    digit = digit.resize((max(1, round(digit.width * scale)), max(1, round(digit.height * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return canvas


def list_models():
    """models 폴더의 *.json(학습 스크립트가 만든 정보 파일)을 읽어 모델 목록 생성"""
    metas = []
    for path in sorted(MODEL_DIR.glob("*.json")):     # model1_, model2_, model3_ 순서로 정렬됨
        with open(path, encoding="utf-8") as f:
            metas.append(json.load(f))
    return metas


@st.cache_resource
def load_model(weights_file):
    """가중치 파일별로 한 번만 불러와 캐시"""
    model = SimpleClassifier()
    model.load_state_dict(torch.load(MODEL_DIR / weights_file, map_location="cpu", weights_only=True))
    return model.eval()


@st.cache_resource
def load_mnist_test():
    return datasets.MNIST(DATA_DIR, train=False, download=True)


@st.cache_resource
def load_qmnist_test50k():
    return datasets.QMNIST(DATA_DIR, what="test50k", download=True)


def predict(model, image):
    """28x28 PIL 이미지 → 클래스별 확률(numpy)"""
    with torch.no_grad():
        return torch.softmax(model(to_tensor(image).unsqueeze(0)), dim=1)[0].numpy()


st.title("MNIST 손글씨 숫자 분류기")

# ---------------- 모델 선택 ----------------
metas = list_models()
if not metas:
    st.error("models 폴더에 모델 정보(.json)가 없습니다. 먼저 `python train_models.py`를 실행하세요.")
    st.stop()

meta = st.selectbox("모델 선택", metas, format_func=lambda m: m["name"])
st.caption(meta["description"])
model = load_model(meta["weights"])

# 모든 모델의 평가 결과를 한 표로 비교 (같은 평가셋, 같은 기준)
with st.expander("모델별 평가 결과 비교"):
    rows = []
    for m in metas:
        row = {"모델": m["name"]}
        for set_name, r in m["metrics"].items():
            row[f"{set_name} 정확도"] = f"{r['accuracy']:.2%}"
            row[f"{set_name} 1→7"] = r["1->7"]
            row[f"{set_name} 7→1"] = r["7->1"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("모델"))

# ---------------- 입력 ----------------
source = st.radio("입력 방식", ["직접 그리기", "MNIST 테스트 샘플", "QMNIST 새 테스트 샘플", "이미지 업로드"],
                  horizontal=True)
image, label = None, None

if source == "직접 그리기":
    st.caption("검은 칸에 숫자 하나를 크게 그려 보세요. 휴지통 아이콘으로 지울 수 있어요.")
    result = st_canvas(                               # 캔버스는 MNIST처럼 검은 배경·흰 글씨 유지
        stroke_width=18, stroke_color="#FFFFFF", background_color="#000000",
        width=280, height=280, drawing_mode="freedraw", key="canvas",
        return_image_data=True,
    )
    if result.image_data is not None:
        drawn = Image.fromarray(result.image_data[:, :, 0].astype(np.uint8))   # 흰 글씨 채널만 사용
        image = to_mnist_style(drawn)

elif source == "MNIST 테스트 샘플":
    idx = st.number_input("샘플 번호 (0~9999)", 0, 9999, 0)
    image, label = load_mnist_test()[idx]

elif source == "QMNIST 새 테스트 샘플":
    idx = st.number_input("샘플 번호 (0~49999)", 0, 49999, 0)
    image, label = load_qmnist_test50k()[idx]

else:
    file = st.file_uploader("숫자 이미지", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file).convert("L")
        if st.checkbox("흰 배경에 검은 글씨면 체크 (색 반전)", value=True):
            img = ImageOps.invert(img)
        image = to_mnist_style(img)

# ---------------- 예측 ----------------
if image is not None:
    col1, col2 = st.columns(2)
    col1.image(image.resize((140, 140), Image.NEAREST), caption="모델이 보는 28x28 이미지")
    probs = predict(model, image)
    pred = int(probs.argmax())
    col2.metric("예측", pred, f"확신도 {probs[pred]:.1%}")
    if label is not None:
        col2.write(f"정답: **{label}** {'✅' if pred == label else '❌'}")
    st.bar_chart({"확률": probs}, x_label="숫자", y_label="확률", color="#2E8B57")

    # 같은 이미지에 대한 모든 모델의 예측을 나란히 비교
    if st.checkbox("모든 모델 예측 비교"):
        compare = []
        for m in metas:
            p = predict(load_model(m["weights"]), image)
            compare.append({"모델": m["name"], "예측": int(p.argmax()),
                            "확신도": f"{p.max():.1%}", "1의 확률": f"{p[1]:.1%}", "7의 확률": f"{p[7]:.1%}"})
        st.dataframe(pd.DataFrame(compare).set_index("모델"))
