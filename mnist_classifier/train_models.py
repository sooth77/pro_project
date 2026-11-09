# 학습 스크립트: 모델 1(기존), 모델 2(모델 1 + 추가 학습), 모델 3(QMNIST)을 만들고 같은 기준으로 평가
# 실행: mnist_app 폴더에서  python train_models.py
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_common import (DATA_DIR, MEAN, MODEL_DIR, STD, SimpleClassifier,
                          evaluate, pick_device, to_tensor)

# ---------------- 실험 설정 (결과 JSON에 그대로 기록됨) ----------------
SEED = 42
BASELINE_WEIGHTS = "mnist_state_dict.pth"   # 기존 가중치 (models 폴더 안)

M2_EXTRA_EPOCHS = 5      # 모델 2: 모델 1에 추가로 학습할 에폭 수
M2_LR = 1e-4             # 모델 2: 이미 학습된 모델이라 작은 학습률로 이어서 학습
M2_AUGMENT = False       # True: 회전·이동·크기·기울임 증강을 넣어 추가 학습 (1·7 개선용)

M3_EPOCHS = 5            # 모델 3: QMNIST 학습셋으로 처음부터 학습할 에폭 수
M3_LR = 1e-3
BATCH_SIZE = 128
# ---------------------------------------------------------------------

# 재현성을 위해 난수 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = pick_device()
print("장치:", device)

# 증강 전처리 (M2_AUGMENT=True일 때만 사용)
augment = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 모든 모델을 같은 평가셋으로 비교 (통제 변수)
EVAL_SETS = {
    "MNIST 테스트 (1만)": datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor),
    "QMNIST 새 테스트 (5만)": datasets.QMNIST(DATA_DIR, what="test50k", download=True, transform=to_tensor),
}


def train(model, dataset, epochs, lr):
    """분류 학습 루프 (Adam + 교차 엔트로피)"""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        print(f"  epoch {ep + 1}/{epochs}  loss {total / n:.4f}  ({time.time() - t0:.0f}초)")
    model.eval()
    return model


def save(model, key, name, description, train_config, weights_file=None):
    """가중치 저장 + 평가 결과와 실험 조건을 JSON으로 저장 (앱이 이 JSON으로 목록을 만듦)"""
    MODEL_DIR.mkdir(exist_ok=True)
    if weights_file is None:                       # 새로 학습한 모델만 가중치를 새로 저장
        weights_file = f"{key}.pth"
        torch.save(model.state_dict(), MODEL_DIR / weights_file)
    metrics = {k: evaluate(model, ds, device) for k, ds in EVAL_SETS.items()}
    meta = {
        "key": key, "name": name, "description": description,
        "weights": weights_file, "train_config": train_config, "metrics": metrics,
        "conditions": {"torch": torch.__version__, "device": str(device), "seed": SEED},
    }
    with open(MODEL_DIR / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[{name}] 저장")
    for k, m in metrics.items():
        print(f"  {k}: 정확도 {m['accuracy']:.2%}  1→7 {m['1->7']}건  7→1 {m['7->1']}건")


def load_baseline():
    """기존 가중치를 불러옴"""
    model = SimpleClassifier().to(device)
    state = torch.load(MODEL_DIR / BASELINE_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


if __name__ == "__main__":
    # 모델 1: 기존 모델을 같은 평가셋으로 측정 (비교 기준선, 가중치는 그대로 사용)
    print("\n=== 모델 1: 기존 모델 (baseline) 평가 ===")
    save(load_baseline(), "model1_baseline", "모델 1: 기존 모델 (baseline)",
         "처음 학습한 MNIST 모델. 다른 모델의 비교 기준", {"source": BASELINE_WEIGHTS},
         weights_file=BASELINE_WEIGHTS)

    # 모델 2: 모델 1에서 이어서 MNIST 학습셋으로 추가 학습
    print(f"\n=== 모델 2: 모델 1 + {M2_EXTRA_EPOCHS}에폭 추가 학습 (증강 {M2_AUGMENT}) ===")
    m2_data = datasets.MNIST(DATA_DIR, train=True, download=True,
                             transform=augment if M2_AUGMENT else to_tensor)
    m2 = train(load_baseline(), m2_data, M2_EXTRA_EPOCHS, M2_LR)
    save(m2, "model2_mnist_more", "모델 2: 모델 1 + 추가 학습",
         f"모델 1 가중치에서 MNIST로 {M2_EXTRA_EPOCHS}에폭 더 학습 (증강 {'사용' if M2_AUGMENT else '없음'})",
         {"init": BASELINE_WEIGHTS, "data": "MNIST train", "epochs": M2_EXTRA_EPOCHS,
          "lr": M2_LR, "augment": M2_AUGMENT, "batch_size": BATCH_SIZE})

    # 모델 3: QMNIST 학습셋으로 처음부터 학습
    print(f"\n=== 모델 3: QMNIST로 처음부터 {M3_EPOCHS}에폭 학습 ===")
    torch.manual_seed(SEED)                         # 모델 3의 초기 가중치도 고정
    m3_data = datasets.QMNIST(DATA_DIR, what="train", download=True, transform=to_tensor)
    m3 = train(SimpleClassifier().to(device), m3_data, M3_EPOCHS, M3_LR)
    save(m3, "model3_qmnist", "모델 3: QMNIST 학습",
         f"QMNIST 학습셋(6만)으로 처음부터 {M3_EPOCHS}에폭 학습",
         {"init": "random", "data": "QMNIST train", "epochs": M3_EPOCHS,
          "lr": M3_LR, "augment": False, "batch_size": BATCH_SIZE})
