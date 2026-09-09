"""학습 전에 데이터를 눈으로 확인한다.

    python tools/preview.py

클래스별 샘플 격자와 분포 그래프를 outputs/ 에 저장한다.
데이터를 안 보고 학습부터 돌리면, 성능이 안 나올 때 원인이 모델인지
데이터인지 구분할 수 없다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots  # noqa: E402
from src.data import CLASS_NAMES, available_sizes, load_split  # noqa: E402


def main() -> int:
    data_root = ROOT / "data"
    sizes = available_sizes(data_root)
    if not sizes:
        print(f"{data_root} 에 데이터가 없습니다.")
        return 1

    size = max(sizes)  # 가장 좋은 해상도로 미리보기
    out = ROOT / "outputs"
    print(f"{size}px 데이터로 미리보기 생성")

    train_imgs, train_lbls = load_split(data_root, size, "train")
    plots.sample_grid(train_imgs, train_lbls, CLASS_NAMES, out / "samples.png")
    print(f"  {out / 'samples.png'}")

    import numpy as np

    counts = {}
    for split in ("train", "val", "test"):
        _, lbls = load_split(data_root, size, split)
        counts[split] = np.bincount(lbls.reshape(-1), minlength=len(CLASS_NAMES))
    plots.class_distribution(counts, CLASS_NAMES, out / "class_distribution.png")
    print(f"  {out / 'class_distribution.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
