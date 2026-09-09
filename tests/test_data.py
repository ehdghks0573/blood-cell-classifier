"""데이터 파이프라인 검증.

이미지가 없어도 도는 부분(변환, Dataset, 경로 규칙)을 먼저 검증한다.
실제 npz 가 필요한 테스트는 파일이 없으면 건너뛴다.
"""

import numpy as np
import pytest
import torch

from src.data import (
    CLASS_NAMES,
    NUM_CLASSES,
    BloodDataset,
    available_sizes,
    build_transform,
    denormalize,
    load_split,
    npz_path,
)

DATA_ROOT = "data"


def fake_images(n=20, size=28):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def fake_labels(n=20):
    return np.arange(n).reshape(-1, 1) % NUM_CLASSES


def test_class_names_count():
    assert len(CLASS_NAMES) == NUM_CLASSES == 8


def test_npz_path_naming():
    """28px 만 접미사가 없다 (medmnist 규칙)."""
    assert npz_path("data", 28).name == "bloodmnist.npz"
    assert npz_path("data", 224).name == "bloodmnist_224.npz"


def test_dataset_length_and_item():
    ds = BloodDataset(fake_images(), fake_labels(), build_transform(64, train=False))
    assert len(ds) == 20
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < NUM_CLASSES


def test_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        BloodDataset(fake_images(20), fake_labels(10))


def test_labels_are_flattened():
    """npz 라벨은 (N,1) 인데 손실 함수는 (N,) 을 받는다."""
    ds = BloodDataset(fake_images(), fake_labels())
    assert ds.labels.ndim == 1
    assert ds.labels.dtype == np.int64


def test_class_counts():
    ds = BloodDataset(fake_images(16), fake_labels(16))
    counts = ds.class_counts()
    assert counts.shape == (NUM_CLASSES,)
    assert counts.sum() == 16


def test_transform_resizes_to_input_size():
    """28px 데이터를 224px 로 늘려 사전학습 모델에 넣을 수 있어야 한다."""
    t = build_transform(224, train=False)
    out = t(fake_images(1, 28)[0])
    assert out.shape == (3, 224, 224)


def test_train_transform_is_random():
    """증강이 실제로 적용되는지 — 같은 이미지를 두 번 넣으면 달라야 한다."""
    torch.manual_seed(0)
    t = build_transform(64, train=True)
    img = fake_images(1)[0]
    assert not torch.allclose(t(img), t(img))


def test_eval_transform_is_deterministic():
    t = build_transform(64, train=False)
    img = fake_images(1)[0]
    assert torch.allclose(t(img), t(img))


def test_denormalize_returns_unit_range():
    t = build_transform(32, train=False)
    out = denormalize(t(fake_images(1)[0]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_missing_size_raises_with_guidance():
    with pytest.raises(FileNotFoundError, match="사용 가능한 해상도"):
        load_split(DATA_ROOT, 999, "train")


@pytest.mark.skipif(not available_sizes(DATA_ROOT), reason="내려받은 데이터가 없음")
def test_real_split_shapes():
    size = available_sizes(DATA_ROOT)[0]
    imgs, lbls = load_split(DATA_ROOT, size, "train")
    assert imgs.ndim == 4 and imgs.shape[3] == 3
    assert imgs.dtype == np.uint8
    assert len(imgs) == len(lbls)
    assert set(np.unique(lbls)) <= set(range(NUM_CLASSES))
