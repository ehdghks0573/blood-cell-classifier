"""BloodMNIST 를 내려받고 구조와 클래스 분포를 확인한다.

224px 은 최종 학습용, 64px 은 빠른 실험용으로 둘 다 받는다.
코드를 고칠 때마다 224px 로 돌리면 한 번에 몇 분씩 날아간다.
"""

import sys
from collections import Counter
from pathlib import Path

import medmnist
from medmnist import INFO

FLAG = "bloodmnist"
SIZES = [64, 224]

# medmnist 는 root 폴더를 직접 만들지 않는다. 없으면 RuntimeError 로 죽는다.
ROOT = Path(__file__).resolve().parents[1] / "data"
ROOT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print(f"medmnist {medmnist.__version__}")

    info = INFO[FLAG]
    labels = info["label"]
    print(f"\n데이터셋 : {info['python_class']}")
    print(f"과제     : {info['task']}")
    print(f"채널     : {info['n_channels']}")
    print(f"클래스   : {len(labels)}종")
    for k, v in labels.items():
        print(f"  {k}  {v}")

    DataClass = getattr(medmnist, info["python_class"])

    for size in SIZES:
        print(f"\n{'=' * 60}\n{size}px 내려받는 중...")
        splits = {}
        for split in ("train", "val", "test"):
            ds = DataClass(split=split, download=True, size=size, root=str(ROOT))
            splits[split] = ds
            print(f"  {split:5s} {len(ds):6,d}장  이미지 {ds.imgs.shape}  라벨 {ds.labels.shape}")

        train = splits["train"]
        print(f"\n  dtype={train.imgs.dtype}  값 범위 {train.imgs.min()}~{train.imgs.max()}")

        # 클래스 분포 — 불균형이 심하면 손실 함수와 평가 지표를 처음부터 그에 맞춰야 한다
        print(f"\n  클래스 분포:")
        print(f"  {'클래스':38s} {'train':>7s} {'val':>6s} {'test':>6s} {'비율':>7s}")
        counts = {s: Counter(splits[s].labels.flatten().tolist()) for s in splits}
        n_train = len(train)
        for k in sorted(labels):
            i = int(k)
            tr = counts["train"][i]
            print(f"  {labels[k][:36]:38s} {tr:7,d} {counts['val'][i]:6,d} "
                  f"{counts['test'][i]:6,d} {tr / n_train * 100:6.1f}%")

        freqs = [counts["train"][int(k)] for k in sorted(labels)]
        ratio = max(freqs) / min(freqs)
        print(f"\n  최다/최소 클래스 비율: {ratio:.1f}배", end="  ")
        if ratio < 2:
            print("→ 균형 잡힌 편")
        elif ratio < 5:
            print("→ 약한 불균형. macro F1 로 평가할 것")
        else:
            print("→ 불균형 심함. 클래스 가중치나 샘플링 필요")

    return 0


if __name__ == "__main__":
    sys.exit(main())
