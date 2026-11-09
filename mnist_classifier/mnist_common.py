# 공통 모듈: 모델 구조, 전처리, 평가 함수 (학습 스크립트와 Streamlit 앱이 함께 사용)
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

BASE = Path(__file__).resolve().parent          # mnist_app 폴더
MODEL_DIR = BASE / "models"                     # 가중치(.pth)와 정보(.json) 저장 위치
DATA_DIR = BASE / "data"                        # 데이터셋 다운로드 위치
MEAN, STD = (0.1307,), (0.3081,)                # MNIST 정규화 값 (QMNIST도 같은 값 사용)


class SimpleClassifier(nn.Module):
    """CNN 2층 + 완전연결 2층. 기존 가중치를 그대로 불러와야 하므로 구조를 바꾸지 않음"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# 평가·추론용 전처리: PIL 이미지 → 텐서 → 정규화
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def pick_device():
    """CUDA → Apple GPU(MPS) → CPU 순서로 사용 가능한 장치 선택"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=1000):
    """정확도, 1↔7 혼동 건수, 10x10 혼동 행렬 계산"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    conf = torch.zeros(10, 10, dtype=torch.long)      # conf[정답, 예측]
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        conf += torch.bincount(y * 10 + pred, minlength=100).view(10, 10)
    total = int(conf.sum())
    return {
        "accuracy": float(conf.trace()) / total,
        "n": total,
        "1->7": int(conf[1, 7]),                      # 정답 1을 7로 틀린 수
        "7->1": int(conf[7, 1]),                      # 정답 7을 1로 틀린 수
        "confusion": conf.tolist(),
    }
