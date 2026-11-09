# 증강 학습 스크립트: 모델 2 → 모델 4, 모델 3 → 모델 5 (각각 증강을 넣어 5에폭 추가 학습)
# 모델 1·2·3은 다시 학습하지 않음. train_models.py의 학습·저장·평가 함수와 증강 설정을 그대로 재사용
# 실행: mnist_app 폴더에서  python train_aug.py
import torch
from torchvision import datasets

from mnist_common import DATA_DIR, MODEL_DIR, SimpleClassifier
from train_models import BATCH_SIZE, SEED, augment, device, save, train  # 같은 함수·같은 평가셋 사용

# ---------------- 실험 설정 ----------------
AUG_EPOCHS = 5           # 모델 4·5 모두 5에폭 추가 학습
AUG_LR = 1e-4            # 이미 학습된 모델에서 이어 가므로 작은 학습률
# 증강 내용: 회전 ±10°, 이동 ±10%, 크기 0.9~1.1배, 기울임 ±10° (train_models.py의 augment)
# -------------------------------------------


def load_weights(weights_file):
    """models 폴더의 가중치를 불러옴. 없으면 무엇을 먼저 실행해야 하는지 알려 줌"""
    path = MODEL_DIR / weights_file
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 train_models.py로 모델 2·3을 만들어 주세요.")
    model = SimpleClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


if __name__ == "__main__":
    # 모델 4: 모델 2에서 이어서, 증강한 MNIST로 5에폭
    print(f"\n=== 모델 4: 모델 2 + 증강 {AUG_EPOCHS}에폭 ===")
    torch.manual_seed(SEED)                         # 섞는 순서와 증강의 무작위성 고정
    m4_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=augment)
    m4 = train(load_weights("model2_mnist_more.pth"), m4_data, AUG_EPOCHS, AUG_LR)
    save(m4, "model4_mnist_aug", "모델 4: 모델 2 + 증강 학습",
         f"모델 2 가중치에서 증강한 MNIST로 {AUG_EPOCHS}에폭 더 학습",
         {"init": "model2_mnist_more.pth", "data": "MNIST train", "epochs": AUG_EPOCHS,
          "lr": AUG_LR, "augment": True, "batch_size": BATCH_SIZE})

    # 모델 5: 모델 3에서 이어서, 증강한 QMNIST로 5에폭
    print(f"\n=== 모델 5: 모델 3 + 증강 {AUG_EPOCHS}에폭 ===")
    torch.manual_seed(SEED)
    m5_data = datasets.QMNIST(DATA_DIR, what="train", download=True, transform=augment)
    m5 = train(load_weights("model3_qmnist.pth"), m5_data, AUG_EPOCHS, AUG_LR)
    save(m5, "model5_qmnist_aug", "모델 5: 모델 3 + 증강 학습",
         f"모델 3 가중치에서 증강한 QMNIST로 {AUG_EPOCHS}에폭 더 학습",
         {"init": "model3_qmnist.pth", "data": "QMNIST train", "epochs": AUG_EPOCHS,
          "lr": AUG_LR, "augment": True, "batch_size": BATCH_SIZE})
