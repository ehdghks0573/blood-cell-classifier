"""여러 모델이 **같은 이미지를** 틀리는가 (S5 의 결정적 증거).

    python tools/consensus_errors.py --runs res112_resnet18 res112_seed43 baseline_resnet18_28px

눈으로 보는 검증에는 한계가 있다. 사람이 "이건 라벨이 이상한데" 하고 느껴도
그건 인상이지 증거가 아니다. 숫자로 물을 수 있는 형태로 바꾸면 이렇게 된다.

**서로 독립적으로 학습한 모델들이 같은 이미지를, 같은 방향으로, 확신에 차서
틀리는가?**

- 모델마다 다른 이미지를 틀린다면 → 학습의 우연(초기화·증강 순서)이 원인이다.
  모델을 더 잘 만들면 줄어든다.
- 모델이 달라도 **같은 이미지**를 **같은 클래스로** 틀린다면 → 원인은 모델이
  아니라 그 이미지에 있다. 라벨이 의심스럽거나, 그 이미지가 정말로 두 클래스
  사이에 있다는 뜻이다.

기준선이 필요하다. 오분류가 우연히 겹칠 확률은 각 모델의 오류율의 곱이다.
83/3421 인 모델 세 개면 우연한 3중 겹침은 3421 × (0.024)^3 ≈ 0.05장, 즉
**거의 0장이어야 한다.** 실제로 수십 장이 겹친다면 우연이 아니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    BloodDataset,
    build_transform,
    load_split,
)
from src.engine import predict  # noqa: E402
from src.model import build_model  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="여러 모델의 오분류가 겹치는지 본다")
    p.add_argument("--runs", nargs="+", required=True, help="runs/ 아래 실행 이름들")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--confident", type=float, default=0.90,
                   help="이 확률 이상으로 틀린 것을 '확신에 차서 틀렸다'로 본다")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save", help="결과 JSON 경로 (기본: docs 에 저장 안 함)")
    return p.parse_args(argv)


def predictions_for(run: str, data_root: str, size: int, split: str, device):
    run_dir = ROOT / "runs" / run
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)

    model = build_model(ckpt["arch"], NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    imgs, lbls = load_split(data_root, size, split)
    dataset = BloodDataset(imgs, lbls, build_transform(ckpt["input_size"], train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_pred, probs, _ = predict(model, loader, device)
    return y_true, y_pred, probs, ckpt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.runs) < 2:
        print("모델이 둘 이상 필요합니다.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    preds, probs_all, truth = {}, {}, None
    for run in args.runs:
        y_true, y_pred, probs, ckpt = predictions_for(
            run, args.data_root, args.size, args.split, device)
        truth = y_true if truth is None else truth
        preds[run] = y_pred
        probs_all[run] = probs
        n_wrong = int((y_pred != y_true).sum())
        print(f"{run:26s} 입력 {ckpt['input_size']}px · 오분류 {n_wrong}장 "
              f"({n_wrong / len(y_true):.2%})")

    n = len(truth)
    wrong = {r: (preds[r] != truth) for r in args.runs}
    print()

    # 우연히 겹칠 기댓값 — 각 모델의 오류율이 독립이라면
    rates = [wrong[r].mean() for r in args.runs]
    expected_all = n * float(np.prod(rates))

    all_wrong = np.ones(n, dtype=bool)
    for r in args.runs:
        all_wrong &= wrong[r]

    print(f"모든 모델이 틀린 장   {int(all_wrong.sum())}장")
    print(f"우연이라면 기대값     {expected_all:.2f}장   "
          f"← {int(all_wrong.sum()) / max(expected_all, 1e-9):.0f}배")
    print()

    print("두 모델씩 겹침")
    for a, b in combinations(args.runs, 2):
        both = int((wrong[a] & wrong[b]).sum())
        exp = n * float(rates[args.runs.index(a)] * rates[args.runs.index(b)])
        print(f"  {a[:22]:<24s} ∩ {b[:22]:<24s} {both:3d}장 (기대 {exp:.1f})")
    print()

    # 같은 방향으로 틀렸는가 — 틀린 것만으로는 부족하다.
    idx = np.flatnonzero(all_wrong)
    same_pred = np.array([len({int(preds[r][i]) for r in args.runs}) == 1 for i in idx],
                         dtype=bool) if idx.size else np.zeros(0, dtype=bool)

    conf = np.array([min(float(probs_all[r][i, preds[r][i]]) for r in args.runs)
                     for i in idx]) if idx.size else np.zeros(0)
    confident = same_pred & (conf >= args.confident)

    print(f"그중 같은 클래스로 틀린 장        {int(same_pred.sum())}장")
    print(f"그중 전부 {args.confident:.0%} 이상 확신한 장   {int(confident.sum())}장")
    print()

    rows = []
    for i, s, c in zip(idx, same_pred, conf):
        if not s:
            continue
        t, p = int(truth[i]), int(preds[args.runs[0]][i])
        rows.append({"index": int(i), "true": CLASS_NAMES[t], "pred": CLASS_NAMES[p],
                     "min_confidence": round(float(c), 4)})
    rows.sort(key=lambda r: -r["min_confidence"])

    print(f"모든 모델이 같은 방향으로 틀린 장 — 확신 높은 순 (상위 15)")
    for r in rows[:15]:
        print(f"  #{r['index']:<5d} {r['true'][:22]:<24s} → {r['pred'][:22]:<24s} "
              f"최소 확신 {r['min_confidence']:.3f}")

    pairs: dict[str, int] = {}
    for r in rows:
        pairs[f"{r['true']} → {r['pred']}"] = pairs.get(f"{r['true']} → {r['pred']}", 0) + 1
    print("\n쌍별 집계")
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<52s} {v}장")

    if args.save:
        out = {
            "runs": args.runs,
            "split": args.split,
            "n": int(n),
            "error_rates": [round(float(x), 5) for x in rates],
            "all_wrong": int(all_wrong.sum()),
            "expected_by_chance": round(expected_all, 3),
            "same_direction": int(same_pred.sum()),
            "confident": int(confident.sum()),
            "cases": rows,
            "pairs": pairs,
        }
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
