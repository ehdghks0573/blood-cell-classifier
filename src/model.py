"""모델 구성.

직접 설계하지 않고 ImageNet 사전학습 모델의 마지막 층만 8클래스로 바꾼다.
데이터가 1.2만 장뿐이라 밑바닥부터 학습하면 과적합이 심하고, 전이학습이
같은 시간에 훨씬 나은 성능을 낸다.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

#: 지원 아키텍처 -> (생성 함수, 가중치 열거형)
ARCHS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def build_model(arch: str = "resnet18", num_classes: int = 8, pretrained: bool = True):
    """분류기 헤드를 num_classes 로 교체한 모델을 만든다."""
    if arch not in ARCHS:
        raise ValueError(f"모르는 아키텍처: {arch}. 가능한 값: {list(ARCHS)}")

    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # efficientnet
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def target_layer(model, arch: str = "resnet18"):
    """Grad-CAM 이 붙을 마지막 conv 블록.

    여기의 활성화와 기울기로 히트맵을 만든다. 마지막 conv 층이라야
    의미 있는 공간 정보와 클래스 정보를 동시에 갖는다.
    """
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"모르는 아키텍처: {arch}")


def finer_layer(model, arch: str = "resnet18"):
    """`target_layer` 보다 한 단계 앞, 공간 해상도가 두 배인 블록.

    112px 입력에서 마지막 블록은 4×4 라 히트맵이 덩어리 하나로 뭉친다.
    7×7 에서도 같은 결론이 나오는지 확인하는 용도다 (`tools/confidence_cam.py`).
    """
    if arch.startswith("resnet"):
        return model.layer3[-1]
    if arch.startswith("efficientnet"):
        return model.features[5]
    raise ValueError(f"모르는 아키텍처: {arch}")


def count_parameters(model) -> tuple[int, int]:
    """(전체, 학습 대상) 파라미터 수."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
