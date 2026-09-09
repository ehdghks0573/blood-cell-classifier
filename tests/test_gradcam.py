"""Grad-CAM 검증.

작은 모델로 CPU 에서 돌려, 히트맵의 모양·값 범위·후크 해제를 확인한다.
학습된 체크포인트가 없어도 도는 테스트다.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.gradcam import GradCAM, overlay


class TinyNet(nn.Module):
    """마지막 conv 층을 가진 최소 모델."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.pool(x).flatten(1))


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = TinyNet()
    return model, model.features[-2]  # 마지막 Conv2d


def test_heatmap_matches_input_size(setup):
    model, layer = setup
    x = torch.randn(1, 3, 32, 32)
    with GradCAM(model, layer) as cam:
        heat, _, _ = cam(x)
    assert heat.shape == (32, 32)


def test_heatmap_is_normalised(setup):
    model, layer = setup
    with GradCAM(model, layer) as cam:
        heat, _, _ = cam(torch.randn(1, 3, 32, 32))
    assert heat.min() >= 0.0
    assert heat.max() <= 1.0


def test_probabilities_sum_to_one(setup):
    model, layer = setup
    with GradCAM(model, layer) as cam:
        _, _, probs = cam(torch.randn(1, 3, 32, 32))
    assert probs.shape == (8,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_defaults_to_predicted_class(setup):
    model, layer = setup
    x = torch.randn(1, 3, 32, 32)
    with GradCAM(model, layer) as cam:
        _, chosen, probs = cam(x)
    assert chosen == int(np.argmax(probs))


def test_explicit_class_is_respected(setup):
    model, layer = setup
    with GradCAM(model, layer) as cam:
        _, chosen, _ = cam(torch.randn(1, 3, 32, 32), class_index=3)
    assert chosen == 3


def test_rejects_batch_input(setup):
    """한 장씩만 설명한다. 배치를 넣으면 기울기가 섞인다."""
    model, layer = setup
    with GradCAM(model, layer) as cam:
        with pytest.raises(ValueError, match="1, C, H, W"):
            cam(torch.randn(4, 3, 32, 32))


def test_hooks_are_removed_on_exit(setup):
    """후크를 안 떼면 이후 추론에서도 계속 붙어 메모리를 잡아먹는다."""
    model, layer = setup
    before = len(layer._forward_hooks)
    with GradCAM(model, layer) as cam:
        assert len(layer._forward_hooks) == before + 1
        cam(torch.randn(1, 3, 32, 32))
    assert len(layer._forward_hooks) == before


def test_model_returns_to_original_mode(setup):
    """설명 중에 eval 로 바꿨다면 원래 모드로 되돌려야 한다."""
    model, layer = setup
    model.train()
    with GradCAM(model, layer) as cam:
        cam(torch.randn(1, 3, 32, 32))
    assert model.training


def test_overlay_shape_and_range():
    img = np.random.rand(3, 16, 16).astype(np.float32)
    heat = np.random.rand(16, 16).astype(np.float32)
    out = overlay(img, heat)
    assert out.shape == (16, 16, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_overlay_leaves_cold_regions_untouched():
    """히트맵이 0 인 곳은 원본이 그대로 보여야 한다."""
    img = np.full((3, 8, 8), 0.5, dtype=np.float32)
    out = overlay(img, np.zeros((8, 8), dtype=np.float32))
    assert np.allclose(out, 0.5)
