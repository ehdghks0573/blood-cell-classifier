"""학습 병목이 어디인지 측정한다.

에폭 시간이 기대보다 길 때 추측으로 num_workers 를 올리면 오히려 느려질 수
있다(4코어 CPU 에서 실제로 그랬다). 데이터 로딩과 GPU 연산을 분리해서 재면
어느 쪽이 병목인지 바로 보인다.

    python tools/bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.console import make_output_encodable  # noqa: E402
from src.data import NUM_CLASSES, BloodDataset, build_transform, load_split  # noqa: E402
from src.model import build_model  # noqa: E402

BATCHES = 40
BATCH_SIZE = 32


def bench_loader(root, size, input_size, num_workers, transform_at_native=False) -> float:
    """DataLoader 만 돌려 배치당 시간을 잰다 (GPU 연산 없음)."""
    imgs, lbls = load_split(root, size, "train")
    tf = build_transform(input_size, train=True)
    ds = BloodDataset(imgs, lbls, tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    it = iter(loader)
    for _ in range(3):  # 워밍업 (워커 기동 비용 제외)
        next(it)

    t0 = time.perf_counter()
    for _ in range(BATCHES):
        next(it)
    return (time.perf_counter() - t0) / BATCHES


def bench_gpu(input_size: int) -> float:
    """합성 텐서로 GPU 학습 스텝만 잰다 (데이터 로딩 없음)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("resnet18", NUM_CLASSES, pretrained=False).to(device)
    x = torch.randn(BATCH_SIZE, 3, input_size, input_size, device=device)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(BATCHES):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / BATCHES


def main() -> int:
    make_output_encodable()
    root = ROOT / "data"
    size, input_size = 28, 224
    steps_per_epoch = 11959 // BATCH_SIZE

    print(f"배치 {BATCH_SIZE} · {size}px → {input_size}px · 에폭당 {steps_per_epoch} 스텝\n")

    gpu = bench_gpu(input_size)
    print(f"GPU 학습 스텝만        : {gpu * 1000:6.1f} ms/배치  "
          f"→ 에폭 {gpu * steps_per_epoch:5.0f}초")

    print()
    for nw in (0, 2, 4):
        t = bench_loader(root, size, input_size, nw)
        print(f"데이터 로딩만 (workers={nw}) : {t * 1000:6.1f} ms/배치  "
              f"→ 에폭 {t * steps_per_epoch:5.0f}초")

    print()
    print("데이터 로딩이 GPU 시간보다 길면 로딩이 병목이다.")
    print("워커를 늘려도 안 줄면 CPU 코어가 부족한 것이므로, 전처리 자체를 줄여야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
