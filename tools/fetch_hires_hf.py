"""224px BloodMNIST 를 HuggingFace 미러에서 받아 공식 npz 모양으로 바꾼다.

    python tools/fetch_hires_hf.py
    python tools/fetch_hires_hf.py --official-prefix 받다_만_공식파일.npz

`tools/fetch_hires.py` 는 공식 Zenodo 주소를 쓴다. 09-17 에 다른 PC 에서 다시
받으려 하니 **분당 6MB** 였다. 1.5GB 면 네 시간이다. 미러
(`danjacobellis/bloodmnist_224`, 723MB)는 초당 6MB 로 2분이면 끝난다.

**미러는 MD5 로 검증할 수 없다.** 파일 형식부터 다르다(parquet + WebP). 그래서
다른 방법으로 같은 데이터인지 잰다.

  1. **번호** — 행마다 `test/17.webp` 같은 원래 번호가 있다. 0 부터 빈틈없이
     이어져야 하고, 그 순서대로 배열을 채운다. parquet 의 행 순서는 믿지 않는다.
  2. **라벨** — 28px 공식 파일의 라벨과 세 분할 모두 정확히 같아야 한다.
  3. **픽셀** — 공식 224px 파일을 받다 만 조각이 있으면(`--official-prefix`),
     거기 들어 있는 앞쪽 학습 이미지와 **픽셀까지 같아야** 한다. 09-17 에 받은
     30MB 조각에서 362장을 꺼내 비교해 362장 모두 일치했다 — 미러의 WebP 는
     무손실이다. 한 칸 어긋나게 짝지으면 평균 차이가 37.4 라 대조군도 갈린다.

그림이 28px 판과 같은 순서인지는 이 도구가 아니라 `tools/check_alignment.py`
가 잰다. 이 도구가 끝나면 그걸 돌린다.

결과는 임시 이름에 다 쓴 뒤에 옮긴다. 쓰다 만 파일이 최종 이름으로 남으면,
기다리던 학습이 반쪽짜리 데이터로 시작한다.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fetch_hires import attempt, human  # noqa: E402
from src.cli import UserError, guard  # noqa: E402
from src.data import load_split, npz_path  # noqa: E402

REPO = "danjacobellis/bloodmnist_224"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/"

#: 우리 분할 이름 → 미러의 parquet 조각들
SHARDS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}
SIZE = 224


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="224px BloodMNIST 를 HF 미러에서 받기")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--cache", default=str(ROOT / "data" / "hf224"),
                   help="parquet 을 받아 둘 폴더")
    p.add_argument("--official-prefix",
                   help="공식 bloodmnist_224.npz 를 받다 만 조각. 있으면 픽셀을 대조한다")
    p.add_argument("--retries", type=int, default=5)
    return p.parse_args(argv)


def index_of(path: str) -> int:
    """`medmnist/bloodmnist/test/17.webp` → 17"""
    return int(path.rsplit("/", 1)[1].split(".")[0])


def download(cache: Path, retries: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name in (n for names in SHARDS.values() for n in names):
        dest = cache / name
        # attempt 는 Range 로 이어받는다. 이미 다 받은 파일이면 416 이 오므로
        # 크기를 따로 확인하지 않고 한 번 더 걸어 본다.
        for i in range(1, retries + 1):
            done, why = attempt(BASE + name, dest, timeout=60)
            if done or "416" in why:
                print(f"  {name}  {human(dest.stat().st_size)}")
                break
            print(f"  {name} 실패 ({why}) — {i}/{retries}")
        else:
            raise UserError(f"{name} 를 받지 못했습니다.")


def decode_split(cache: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image

    rows = []
    for name in SHARDS[split]:
        rows += pq.read_table(cache / name, columns=["path", "label", "image"]).to_pylist()

    order = sorted(range(len(rows)), key=lambda k: index_of(rows[k]["path"]))
    got = [index_of(rows[k]["path"]) for k in order]
    if got != list(range(len(rows))):
        raise UserError(f"[{split}] 번호가 0..{len(rows) - 1} 로 이어지지 않습니다.")

    images = np.empty((len(rows), SIZE, SIZE, 3), dtype=np.uint8)
    labels = np.empty((len(rows), 1), dtype=np.uint8)
    for i, k in enumerate(order):
        img = Image.open(io.BytesIO(rows[k]["image"]["bytes"])).convert("RGB")
        if img.size != (SIZE, SIZE):
            raise UserError(f"[{split}] {i}번이 {img.size} 입니다.")
        images[i] = np.asarray(img)
        labels[i] = rows[k]["label"]
    return images, labels


def official_train_prefix(path: Path) -> np.ndarray:
    """받다 만 공식 npz 에서 앞쪽 학습 이미지를 꺼낸다.

    npz 는 zip 이고 첫 항목이 `train_images.npy` 다. 뒤쪽 목차가 없어도
    첫 항목의 머리와 deflate 흐름은 앞에서부터 풀 수 있다.
    """
    import numpy.lib.format as fmt

    raw = path.read_bytes()
    sig, _, _, method = struct.unpack("<IHHH", raw[:10])
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    name = raw[30:30 + name_len].decode()
    if sig != 0x04034B50 or name != "train_images.npy":
        raise UserError(f"{path} 는 공식 npz 조각이 아닙니다 (첫 항목 {name!r}).")
    body = raw[30 + name_len + extra_len:]
    data = zlib.decompressobj(-15).decompress(body) if method == 8 else body

    f = io.BytesIO(data)
    version = fmt.read_magic(f)
    fmt._read_array_header(f, version)
    start = f.tell()
    n = (len(data) - start) // (SIZE * SIZE * 3)
    return np.frombuffer(data[start:start + n * SIZE * SIZE * 3], np.uint8).reshape(n, SIZE, SIZE, 3)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root, cache = Path(args.data_root), Path(args.cache)
    dest = npz_path(root, SIZE)
    if dest.exists():
        print(f"{dest} 가 이미 있습니다. 다시 만들려면 지우고 실행하세요.")
        return 0
    if not npz_path(root, 28).exists():
        raise UserError("라벨을 대조할 28px 파일이 없습니다. tools/fetch_data.py 를 먼저 돌리세요.")

    print(f"미러: {REPO}")
    download(cache, args.retries)

    arrays = {}
    for split in SHARDS:
        images, labels = decode_split(cache, split)
        _, lo_labels = load_split(root, 28, split)
        same = np.array_equal(labels.reshape(-1), np.asarray(lo_labels).reshape(-1))
        print(f"[{split}] {len(labels):,}장 · 28px 라벨과 일치: {'통과' if same else '실패'}")
        if not same:
            raise UserError(f"[{split}] 라벨이 28px 공식 파일과 다릅니다. 이 미러는 쓸 수 없습니다.")
        arrays[f"{split}_images"], arrays[f"{split}_labels"] = images, labels

    if args.official_prefix:
        ref = official_train_prefix(Path(args.official_prefix))
        ours = arrays["train_images"][:len(ref)]
        exact = int(sum(np.array_equal(a, b) for a, b in zip(ours, ref)))
        control = float(np.abs(ours[1:51].astype(np.int16) - ref[:50].astype(np.int16)).mean())
        print(f"공식 조각 대조: {exact}/{len(ref)} 장 픽셀 일치 · 어긋난 짝 평균 차이 {control:.1f}")
        if exact != len(ref):
            raise UserError("공식 파일과 픽셀이 다릅니다. 이 미러는 쓸 수 없습니다.")

    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(dest)
    print(f"\n{dest} ({human(dest.stat().st_size)})")
    print("다음: python tools/check_alignment.py")
    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
