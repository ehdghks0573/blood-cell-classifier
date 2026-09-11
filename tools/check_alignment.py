"""28px 과 224px 이 **같은 순서의 같은 이미지인가**.

    python tools/check_alignment.py
    python tools/check_alignment.py --samples 200

왜 이걸 따로 재는가. 이 프로젝트의 핵심 결과인 "다섯 모델이 같은 26장을
틀린다"는 **테스트셋 인덱스**로 저장돼 있다(`runs/consensus_errors.json`).
고해상도로 다시 학습해 그 26장이 살아남는지 보려면, 224px 의 3421번이
28px 의 3421번과 같은 사진이어야 한다. 다르면 비교가 통째로 무의미하다.

MedMNIST 는 해상도별 파일이 같은 분할·같은 순서라고 말한다. 말을 믿고
결과를 쓰는 대신 잰다. 세 가지를 본다.

  1. **개수와 라벨** — 분할마다 길이와 라벨 배열이 정확히 일치하는가
  2. **그림 자체** — 224px 를 28px 로 줄이면 28px 판과 같아지는가
  3. **대조군** — 인덱스를 어긋나게 짝지으면 그 차이가 커지는가

3 번이 없으면 2 번은 증거가 못 된다. "혈액 도말 사진끼리는 원래 비슷하다"는
설명이 남기 때문이다. 짝이 맞을 때와 어긋날 때의 차이가 갈려야 인덱스가
의미를 갖는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.console import make_output_encodable  # noqa: E402
from src.data import load_split, npz_path  # noqa: E402

SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="해상도 간 인덱스 정렬 검증")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--low", type=int, default=28)
    p.add_argument("--high", type=int, default=224)
    p.add_argument("--samples", type=int, default=100,
                   help="그림을 비교할 표본 수 (분할마다)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def shrink(img: np.ndarray, size: int) -> np.ndarray:
    """224px 한 장을 28px 로 줄인다. MedMNIST 가 원본을 줄인 것과 같은 방식."""
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def main(argv: list[str] | None = None) -> int:
    make_output_encodable()
    args = parse_args(argv)
    root = Path(args.data_root)

    for size in (args.low, args.high):
        path = npz_path(root, size)
        if not path.exists():
            print(f"{path} 가 없습니다.")
            print(f"  python tools/fetch_hires.py --size {size}")
            return 1

    rng = np.random.default_rng(args.seed)
    ok = True

    print(f"{args.low}px 과 {args.high}px 을 맞춰 봅니다. 표본 {args.samples}장/분할\n")

    for split in SPLITS:
        lo_imgs, lo_lbls = load_split(root, args.low, split)
        hi_imgs, hi_lbls = load_split(root, args.high, split)

        # ── 1. 개수와 라벨 ──
        same_len = len(lo_lbls) == len(hi_lbls)
        same_lbls = same_len and np.array_equal(lo_lbls, hi_lbls)
        mark = "통과" if same_lbls else "실패"
        print(f"[{split}] {len(lo_lbls)} vs {len(hi_lbls)} 장 · 라벨 일치: {mark}")
        if not same_lbls:
            ok = False
            if same_len:
                bad = int((lo_lbls != hi_lbls).sum())
                print(f"  라벨이 다른 자리 {bad}개 — 순서가 다릅니다")
            continue

        # ── 2. 그림 자체 ──
        n = min(args.samples, len(lo_lbls))
        idx = rng.choice(len(lo_lbls), size=n, replace=False)
        matched = [mean_abs_diff(shrink(hi_imgs[i], args.low), lo_imgs[i]) for i in idx]

        # ── 3. 대조군: 한 칸씩 밀어 일부러 어긋나게 짝짓는다 ──
        shifted = np.roll(idx, 1)
        mismatched = [mean_abs_diff(shrink(hi_imgs[j], args.low), lo_imgs[i])
                      for i, j in zip(idx, shifted)]

        m, x = float(np.mean(matched)), float(np.mean(mismatched))
        worst = float(np.max(matched))
        print(f"  픽셀 차이(0~255): 짝이 맞을 때 {m:.2f} (최악 {worst:.2f}) · "
              f"어긋났을 때 {x:.2f}")

        # 짝이 맞은 쪽이 어긋난 쪽보다 뚜렷하게 작아야 인덱스가 의미를 갖는다.
        if m * 3 < x:
            # m 이 0 이면 두 파일이 픽셀까지 같다는 뜻이다 — 배수를 낼 수 없다.
            gap = "완전히 같습니다" if m == 0 else f"{x / m:.0f}배 차이"
            print(f"  → 같은 순서의 같은 이미지입니다 ({gap})")
        else:
            ok = False
            print("  → 구분이 안 됩니다. 인덱스로 두 해상도를 이어 쓸 수 없습니다")

    print()
    if ok:
        print(f"{args.low}px 인덱스를 {args.high}px 에 그대로 쓸 수 있습니다.")
        print("runs/consensus_errors.json 의 26장을 고해상도에서 다시 볼 수 있습니다.")
        return 0
    print("정렬이 깨졌습니다. 해상도 간 인덱스 비교를 하면 안 됩니다.")
    return 1


if __name__ == "__main__":
    from src.cli import guard

    sys.exit(guard(main))
