"""해상도가 다른 두 파일을 인덱스로 이어 써도 되는지 판정하는 규칙.

이 판정이 틀리면 **그 다음 결과가 전부 조용히 무의미해진다.** 224px 에서
다시 학습해 "그 26장이 살아남았다"고 말하려면, 두 파일의 26번이 같은 사진
이어야 한다. 순서가 어긋난 것을 통과시키면, 아무 상관 없는 사진들을 비교해
놓고 결론을 쓰게 된다.

그래서 못박는 것은 두 방향이다 — **맞는 것을 통과시키는가**, 그리고
**어긋난 것을 잡아내는가**. 뒤쪽이 없으면 "항상 통과"와 구분되지 않는다.
"""

import numpy as np
import pytest

from tools.check_alignment import main

SPLITS = ("train", "val", "test")


def write_pair(root, lo_images, hi_images, lo_labels, hi_labels, low=28, high=224):
    """28px/224px npz 한 쌍을 만든다. 분할 세 개가 모두 있어야 도구가 돈다."""
    for size, imgs, lbls in ((low, lo_images, lo_labels), (high, hi_images, hi_labels)):
        name = "bloodmnist.npz" if size == 28 else f"bloodmnist_{size}.npz"
        np.savez(root / name, **{f"{s}_{k}": v for s in SPLITS
                                 for k, v in (("images", imgs), ("labels", lbls))})


def cells(n, size, seed=0):
    """세포 사진 흉내. 장마다 다른 자리에 다른 밝기의 덩어리를 놓는다.

    완전한 난수로 만들면 줄였을 때 전부 회색으로 뭉개져 짝이 맞는지 알 수
    없다. 실제 데이터처럼 **장마다 구분되는 구조**가 있어야 한다.
    """
    rng = np.random.default_rng(seed)
    out = np.full((n, size, size, 3), 40, dtype=np.uint8)
    for i in range(n):
        y, x = rng.integers(0, size // 2, size=2)
        r = size // 3
        out[i, y:y + r, x:x + r] = rng.integers(80, 255, size=3, dtype=np.uint8)
    return out


@pytest.fixture
def matched(tmp_path):
    """올바른 한 쌍 — 224px 를 줄인 것이 28px 다."""
    from PIL import Image
    hi = cells(12, 224)
    lo = np.stack([np.asarray(Image.fromarray(img).resize((28, 28), Image.BILINEAR))
                   for img in hi])
    labels = np.arange(12).reshape(-1, 1) % 8
    write_pair(tmp_path, lo, hi, labels, labels)
    return tmp_path


def run(root, *extra):
    return main(["--data-root", str(root), "--samples", "12", *extra])


def test_matched_pair_passes(matched):
    assert run(matched) == 0


def test_shuffled_images_are_caught(tmp_path, matched):
    """라벨은 그대로 두고 그림만 섞는다 — 라벨 검사만으로는 못 잡는 경우."""
    from PIL import Image
    hi = cells(12, 224, seed=1)
    lo = np.stack([np.asarray(Image.fromarray(img).resize((28, 28), Image.BILINEAR))
                   for img in hi])
    labels = np.arange(12).reshape(-1, 1) % 8
    write_pair(tmp_path, lo[::-1], hi, labels, labels)  # 28px 만 뒤집는다
    assert run(tmp_path) == 1


def test_different_labels_are_caught(tmp_path):
    hi = cells(12, 224, seed=2)
    lo = cells(12, 28, seed=2)
    labels = np.arange(12).reshape(-1, 1) % 8
    write_pair(tmp_path, lo, hi, labels, labels[::-1])
    assert run(tmp_path) == 1


def test_different_lengths_are_caught(tmp_path):
    hi = cells(12, 224, seed=3)
    lo = cells(10, 28, seed=3)
    write_pair(tmp_path, lo, hi,
               np.arange(10).reshape(-1, 1) % 8, np.arange(12).reshape(-1, 1) % 8)
    assert run(tmp_path) == 1


def test_missing_file_is_explained_not_crashed(tmp_path, capsys):
    assert run(tmp_path) == 1
    assert "fetch_hires.py" in capsys.readouterr().out
