"""내려받은 BloodMNIST npz 를 열어 구조와 클래스 분포를 확인한다.

medmnist 의 자동 다운로드는 Zenodo 를 쓰는데 막히는 환경이 있다.
그래서 다운로드와 검사를 분리해, 어떤 경로로 받았든 파일만 있으면 확인되게 한다.

    python tools/inspect_data.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from medmnist import INFO

FLAG = "bloodmnist"
ROOT = Path(__file__).resolve().parents[1] / "data"

# 파일명 규칙: 28px 는 접미사 없음, 나머지는 _<size>
CANDIDATES = [
    ("28px", ROOT / "bloodmnist.npz"),
    ("64px", ROOT / "bloodmnist_64.npz"),
    ("128px", ROOT / "bloodmnist_128.npz"),
    ("224px", ROOT / "bloodmnist_224.npz"),
]


def describe(name: str, path: Path, labels: dict) -> None:
    print(f"\n{'=' * 66}\n{name}  ({path.name}, {path.stat().st_size / 1024**2:.1f} MB)")

    with np.load(path) as z:
        keys = sorted(z.files)
        splits = {}
        for split in ("train", "val", "test"):
            imgs = z[f"{split}_images"]
            lbls = z[f"{split}_labels"]
            splits[split] = (imgs, lbls)
            print(f"  {split:5s} {len(imgs):6,d}장  이미지 {imgs.shape}  라벨 {lbls.shape}")

        tr_imgs = splits["train"][0]
        print(f"\n  dtype={tr_imgs.dtype}  값 범위 {tr_imgs.min()}~{tr_imgs.max()}")
        print(f"  npz 키: {', '.join(keys)}")

    counts = {s: Counter(splits[s][1].flatten().tolist()) for s in splits}
    n_train = len(splits["train"][0])

    print(f"\n  {'클래스':34s} {'train':>7s} {'val':>6s} {'test':>6s} {'비율':>7s}")
    for k in sorted(labels, key=int):
        i = int(k)
        tr = counts["train"][i]
        short = labels[k].split("(")[0].strip()[:32]
        print(f"  {short:34s} {tr:7,d} {counts['val'][i]:6,d} "
              f"{counts['test'][i]:6,d} {tr / n_train * 100:6.1f}%")

    freqs = [counts["train"][int(k)] for k in sorted(labels, key=int)]
    ratio = max(freqs) / min(freqs)
    print(f"\n  최다/최소 비율: {ratio:.1f}배", end="  ")
    if ratio < 2:
        print("→ 비교적 균형")
    elif ratio < 5:
        print("→ 약한 불균형. macro F1 로 평가하고 클래스별 지표를 함께 본다")
    else:
        print("→ 불균형 심함. 클래스 가중치나 가중 샘플링이 필요하다")


def main() -> int:
    labels = INFO[FLAG]["label"]
    print(f"BloodMNIST · {len(labels)}클래스 · {INFO[FLAG]['task']}")

    found = [(n, p) for n, p in CANDIDATES if p.exists()]
    if not found:
        print(f"\n{ROOT} 에 npz 파일이 없습니다.")
        print("tools/fetch_data.py 를 실행하거나, npz 를 직접 넣으세요.")
        return 1

    for name, path in found:
        describe(name, path, labels)

    missing = [n for n, p in CANDIDATES if not p.exists()]
    if missing:
        print(f"\n없는 해상도: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
