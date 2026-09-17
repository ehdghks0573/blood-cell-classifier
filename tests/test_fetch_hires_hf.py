"""HF 미러를 공식 npz 로 바꿀 때 지켜야 하는 규칙.

미러는 MD5 로 검증할 수 없다. 대신 **번호 순서**와 **공식 파일 조각과의 픽셀
대조**로 같은 데이터인지 잰다. 이 둘이 조용히 틀리면 "224px 에서도 살아남았다"는
결론이 엉뚱한 사진들 위에서 쓰인다.
"""

import io

import numpy as np
import pytest

from src.cli import UserError
from tools import fetch_hires_hf as hf


def test_index_of_reads_the_original_number():
    assert hf.index_of("medmnist/bloodmnist/test/17.webp") == 17
    assert hf.index_of("medmnist/bloodmnist/train/0.webp") == 0


def fake_rows(indices, labels):
    """parquet 한 조각의 행 흉내. 장마다 밝기가 달라 순서를 확인할 수 있다."""
    from PIL import Image

    rows = []
    for i, lab in zip(indices, labels):
        img = Image.fromarray(np.full((hf.SIZE, hf.SIZE, 3), i, dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="WEBP", lossless=True)
        rows.append({"path": f"medmnist/bloodmnist/test/{i}.webp",
                     "label": [lab], "image": {"bytes": buf.getvalue(), "path": None}})
    return rows


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


def patch_parquet(monkeypatch, rows):
    import pyarrow.parquet as pq

    monkeypatch.setattr(pq, "read_table", lambda *a, **k: FakeTable(rows))
    monkeypatch.setitem(hf.SHARDS, "test", ["only.parquet"])


def test_decode_orders_by_number_not_by_row(monkeypatch, tmp_path):
    """parquet 행 순서가 섞여 있어도 번호대로 채운다."""
    pytest.importorskip("pyarrow")
    patch_parquet(monkeypatch, fake_rows([2, 0, 1], [7, 5, 6]))

    images, labels = hf.decode_split(tmp_path, "test")

    assert [int(images[i, 0, 0, 0]) for i in range(3)] == [0, 1, 2]
    assert labels.reshape(-1).tolist() == [5, 6, 7]


def test_decode_rejects_a_gap_in_numbers(monkeypatch, tmp_path):
    """번호가 빠져 있으면 뒤쪽 인덱스가 전부 한 칸씩 밀린다 — 멈춰야 한다."""
    pytest.importorskip("pyarrow")
    patch_parquet(monkeypatch, fake_rows([0, 1, 3], [0, 0, 0]))

    with pytest.raises(UserError):
        hf.decode_split(tmp_path, "test")


def truncated_official(tmp_path, images, keep_bytes):
    """공식 npz 를 받다 만 조각을 만든다. 공식 파일은 deflate 압축이다."""
    full = tmp_path / "full.npz"
    np.savez_compressed(full, train_images=images, train_labels=np.zeros((len(images), 1)))
    part = tmp_path / "part.bin"
    part.write_bytes(full.read_bytes()[:keep_bytes])
    return part


def test_official_prefix_recovers_leading_images(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(6, hf.SIZE, hf.SIZE, 3), dtype=np.uint8)
    part = truncated_official(tmp_path, images, keep_bytes=4 * hf.SIZE * hf.SIZE * 3)

    got = hf.official_train_prefix(part)

    assert 1 <= len(got) < len(images)
    assert np.array_equal(got, images[:len(got)])


def test_official_prefix_rejects_other_files(tmp_path):
    other = tmp_path / "other.npz"
    np.savez(other, test_images=np.zeros((1, 2, 2, 3), dtype=np.uint8))

    with pytest.raises(UserError):
        hf.official_train_prefix(other)
