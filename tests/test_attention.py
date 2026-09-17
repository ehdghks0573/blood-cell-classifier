"""히트맵이 세포 위에 떨어지는 비율과, 그 비율로 판정하는 규칙.

두 방향을 다 못박는다 — 세포를 보는 히트맵은 높게, **세포 밖을 보는 히트맵은
낮게** 나와야 한다. 뒤쪽이 없으면 "항상 높게 나오는 숫자"와 구분되지 않는다.
"""

import numpy as np
import pytest

from src import attention as A
from tools.confidence_cam import assign_groups


def blob_image(size=28, center=(14, 14), radius=6):
    """밝은 배경에 어두운 원 하나. 도말 사진의 세포 흉내."""
    yy, xx = np.mgrid[:size, :size]
    inside = np.hypot(yy - center[0], xx - center[1]) <= radius
    img = np.full((size, size, 3), 230, dtype=np.uint8)
    img[inside] = (120, 90, 150)
    return img, inside


def test_otsu_splits_two_levels():
    gray = np.array([10] * 50 + [200] * 50, dtype=np.float64)
    t = A.otsu_threshold(gray)
    assert 10 <= t < 200


def test_cell_mask_marks_the_dark_blob():
    img, inside = blob_image()
    assert np.array_equal(A.cell_mask(img, 28), inside)


def test_cell_mask_scales_to_input_size():
    img, _ = blob_image()
    mask = A.cell_mask(img, 112)
    assert mask.shape == (112, 112)
    assert mask[56, 56] and not mask[0, 0]


def test_inside_ratio_high_on_cell_low_off_cell():
    _, inside = blob_image()
    on = inside.astype(float)
    off = (~inside).astype(float)
    assert A.inside_ratio(on, inside) == pytest.approx(1.0)
    assert A.inside_ratio(off, inside) == pytest.approx(0.0)


def test_inside_ratio_of_flat_heatmap_is_the_mask_area():
    """고른 히트맵은 면적만큼만 들어간다 — 이게 '아무것도 안 보는' 기준선이다."""
    _, inside = blob_image()
    assert A.inside_ratio(np.ones_like(inside, float), inside) == pytest.approx(inside.mean())


def test_inside_ratio_empty_heatmap_is_nan():
    _, inside = blob_image()
    assert np.isnan(A.inside_ratio(np.zeros_like(inside, float), inside))


def test_inside_ratio_rejects_size_mismatch():
    with pytest.raises(ValueError):
        A.inside_ratio(np.ones((4, 4)), np.ones((5, 5), bool))


def test_swapped_mask_control_separates_when_cells_move():
    """세포 자리가 장마다 다르면, 남의 마스크를 댄 값이 뚜렷이 낮아야 한다."""
    _, a = blob_image(center=(7, 7), radius=5)
    _, b = blob_image(center=(21, 21), radius=5)
    heat_a = a.astype(float)
    assert A.inside_ratio(heat_a, a) > A.inside_ratio(heat_a, b) + 0.5


def test_verdict_withholds_when_interval_straddles_zero():
    assert A.verdict(0.01, 0.2) == "크다"
    assert A.verdict(-0.2, -0.01) == "작다"
    assert A.verdict(-0.05, 0.05) == "판정보류"


def test_diff_ci_detects_a_real_gap_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 200)
    b = rng.normal(0.6, 0.05, 200)
    same = rng.normal(0.8, 0.05, 200)
    _, lo, hi = A.diff_ci(a, b)
    assert A.verdict(lo, hi) == "크다"
    _, lo, hi = A.diff_ci(a, same)
    assert A.verdict(lo, hi) == "판정보류"


def test_mean_ci_ignores_nan():
    m, lo, hi = A.mean_ci(np.array([0.5, np.nan, 0.5]))
    assert m == lo == hi == pytest.approx(0.5)


def test_assign_groups_by_confidence_and_correctness():
    y = np.array([0, 0, 0, 1])
    probs = np.array([
        [0.995, 0.005],  # 확신 높은 정답
        [0.60, 0.40],    # 확신 낮은 정답
        [0.95, 0.05],    # 정답이지만 어느 쪽도 아님
        [0.70, 0.30],    # 오답
    ])
    assert assign_groups(y, probs).tolist() == ["high", "low", "", "wrong"]


def test_centered_disk_keeps_area_and_sits_in_the_middle():
    _, off = blob_image(center=(6, 6), radius=4)
    disk = A.centered_disk(off)
    assert disk.sum() == off.sum()
    assert disk[14, 14] and not disk[0, 0]


def test_disk_control_separates_shape_following_from_centre_bias():
    """세포가 구석에 있을 때, 세포를 따라간 히트맵은 가운데 원보다 높아야 한다.
    가운데만 보는 히트맵은 그 반대여야 한다 — 두 방향이 다 갈려야 대조군이다."""
    _, cell = blob_image(center=(7, 7), radius=5)
    follows = cell.astype(float)
    centre = A.centered_disk(cell).astype(float)
    assert A.inside_ratio(follows, cell) > A.inside_ratio(follows, A.centered_disk(cell))
    assert A.inside_ratio(centre, cell) < A.inside_ratio(centre, A.centered_disk(cell))


def test_largest_component_drops_edge_fragments():
    img, cell = blob_image(center=(14, 14), radius=6)
    img[0:2, 0:2] = (120, 90, 150)  # 가장자리에 걸린 어두운 조각
    assert A.cell_mask(img, 28)[0, 0]
    only = A.cell_mask(img, 28, largest=True)
    assert not only[0, 0]
    assert np.array_equal(only, cell)
