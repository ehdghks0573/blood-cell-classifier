"""히트맵이 세포 위에 얼마나 떨어지는가 — Day 4 의 "확신과 주목 위치" 비교용.

Grad-CAM 을 눈으로만 보면 "세포를 보고 있다"는 인상이 남을 뿐 비교가 안 된다.
그래서 한 장마다 숫자 하나로 줄인다.

    세포 위 비율 = (세포 영역 안의 히트맵 합) / (히트맵 전체 합)

**세포 영역은 밝기로 가른다.** 도말 사진은 배경이 밝은 분홍이고 염색된 세포가
어둡다. 28px 원본의 회색조에 Otsu 임계값을 걸면 어두운 쪽이 세포다. 시험셋
500장에서 이 영역은 평균 면적 30%, 중심에서 평균 6.7px(고른 분포라면 10.7px)로
세포가 놓이는 가운데에 모인다.

**이 숫자에는 대조군이 필요하다.** 히트맵이 원래 가운데에 몰리는 모양이라면,
어떤 가운데 마스크를 대도 비율이 높게 나온다. 그래서 같은 히트맵에 **다른
장의 마스크**를 대 본 값을 함께 낸다. 그런데 이 데이터는 세포가 거의 늘
가운데라 남의 마스크도 가운데에 겹쳐 갈리지 않았다. 그래서 **면적이 같은
정중앙 원**을 두 번째 대조군으로 둔다(`centered_disk`). 제 마스크일 때가 이
원보다 뚜렷하게 높아야 "세포를 따라 본다"는 말이 성립한다. 통과하지 못하는
경우가 없는 검사는 검사가 아니다 (`docs/results.md` 4절).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def otsu_threshold(gray: np.ndarray) -> float:
    """0~255 회색조의 Otsu 임계값. 두 무리의 분산 합이 가장 작아지는 자리."""
    hist = np.bincount(np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256)
    hist = hist.astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * levels)
    m0 = s0 / np.where(w0 > 0, w0, 1)
    m1 = (s0[-1] - s0) / np.where(w1 > 0, w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return float(np.argmax(between))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """4-이웃으로 이어진 덩어리 중 가장 큰 것만 남긴다."""
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (int(np.argmax(sizes)) + 1)


def cell_mask(image_hwc: np.ndarray, size: int, largest: bool = False) -> np.ndarray:
    """원본 한 장(uint8 HWC)에서 세포(어두운 쪽) 영역을 `size`×`size` 로 돌려준다.

    임계값은 원본 해상도에서 잡는다. 늘린 뒤에 잡으면 보간으로 생긴 중간 밝기가
    분포를 흐린다.

    `largest` 를 켜면 가장 큰 덩어리만 남긴다. Otsu 는 가장자리에 걸린 적혈구
    조각도 어둡다고 잡는데, 그게 결론을 흔드는지 보려고 둔 선택지다.
    """
    gray = image_hwc.astype(np.float64).mean(axis=-1)
    dark = gray <= otsu_threshold(gray)
    if largest:
        dark = largest_component(dark)
    if dark.shape != (size, size):
        dark = np.asarray(Image.fromarray(dark.astype(np.uint8) * 255)
                          .resize((size, size), Image.NEAREST)) > 0
    return dark


def centered_disk(mask: np.ndarray) -> np.ndarray:
    """`mask` 와 면적이 같고 정중앙에 놓인 원.

    남의 마스크를 대는 대조군만으로는 부족했다. 이 데이터는 세포가 거의 늘
    가운데에 있어서, 남의 마스크도 가운데에 겹친다 — 09-17 첫 실행에서 제
    마스크 0.467, 남의 마스크 0.458 로 갈리지 않았다. 그러면 "세포를 본다"와
    "가운데를 본다"를 구분할 수 없다. 이 원보다 제 마스크가 높아야 히트맵이
    **그 세포의 모양과 자리**를 따라간다고 말할 수 있다.
    """
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2).ravel()
    k = int(mask.sum())
    disk = np.zeros(h * w, dtype=bool)
    disk[np.argsort(dist, kind="stable")[:k]] = True
    return disk.reshape(h, w)


def inside_ratio(heat: np.ndarray, mask: np.ndarray) -> float:
    """히트맵 질량 중 마스크 안에 떨어진 비율. 히트맵이 전부 0 이면 nan."""
    if heat.shape != mask.shape:
        raise ValueError(f"히트맵 {heat.shape} 와 마스크 {mask.shape} 의 크기가 다릅니다")
    total = float(heat.sum())
    if total <= 0:
        return float("nan")
    return float(heat[mask].sum()) / total


def mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """평균과 95% 부트스트랩 구간. nan 은 뺀다. 시드를 고정해 판정이 흔들리지 않게 한다."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
            seed: int = 0) -> tuple[float, float, float]:
    """평균 차이(a − b)와 95% 부트스트랩 구간."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    lo, hi = np.percentile(da - db, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    """구간이 0 을 걸치면 판정하지 않는다. 부등호 하나로 판정하다 뒤집힌 적이 있다."""
    if lo > 0:
        return "크다"
    if hi < 0:
        return "작다"
    return "판정보류"
