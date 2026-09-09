"""저장된 체크포인트를 평가한다.

    python evaluate.py --run baseline_resnet18_28px

학습과 분리해 둔 이유는 두 가지다.
  1) 학습이 끝나기를 기다리지 않고 중간 체크포인트를 평가할 수 있다
  2) 평가 코드를 고쳐도 30분짜리 학습을 다시 돌릴 필요가 없다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import metrics as M  # noqa: E402
from src import plots  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_KO,
    NUM_CLASSES,
    BloodDataset,
    build_transform,
    load_split,
)
from src.engine import predict  # noqa: E402
from src.model import build_model  # noqa: E402

# 성공 기준 (docs/PLAN.md 와 일치시킨다)
S1_MACRO_F1 = 0.96
S2_MIN_CLASS_F1 = 0.90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="체크포인트 평가")
    p.add_argument("--run", required=True, help="runs/ 아래 실행 이름")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save", action="store_true",
                   help="지표와 혼동행렬 이미지를 runs/<run>/ 에 저장한다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = ROOT / "runs" / args.run
    ckpt_path = run_dir / "best.pt"

    if not ckpt_path.exists():
        print(f"{ckpt_path} 가 없습니다. train.py 를 먼저 실행하세요.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(ckpt["arch"], NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    imgs, lbls = load_split(args.data_root, args.size, args.split)
    dataset = BloodDataset(imgs, lbls, build_transform(ckpt["input_size"], train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"실행     : {args.run}")
    print(f"체크포인트: {ckpt['arch']} · 에폭 {ckpt['epoch']} "
          f"· val macro F1 {ckpt['val_macro_f1']:.4f}")
    print(f"평가 대상: {args.split} {len(dataset):,}장 ({args.size}px → {ckpt['input_size']}px)")
    print()

    y_true, y_pred, _, _ = predict(model, loader, device)
    m = M.compute(y_true, y_pred, NUM_CLASSES)
    worst_i, worst_f1 = m.worst_class

    ok = lambda cond: "OK " if cond else "미달"  # noqa: E731
    print(f"accuracy    {m.accuracy:.4f}")
    print(f"macro F1    {m.macro_f1:.4f}   [{ok(m.macro_f1 >= S1_MACRO_F1)}] "
          f"S1 (>= {S1_MACRO_F1})")
    print(f"weighted F1 {m.weighted_f1:.4f}")
    print(f"최저 클래스 {CLASS_NAMES[worst_i]} {worst_f1:.4f}   "
          f"[{ok(worst_f1 >= S2_MIN_CLASS_F1)}] S2 (>= {S2_MIN_CLASS_F1})")
    print()

    print("클래스별 F1")
    order = sorted(range(NUM_CLASSES), key=lambda i: m.per_class_f1[i])
    for i in order:
        bar = "#" * int(m.per_class_f1[i] * 40)
        print(f"  {CLASS_NAMES_KO[i]:<8s} {CLASS_NAMES[i][:24]:<26s} "
              f"{m.per_class_f1[i]:.4f}  {bar}")
    print()

    print(M.report_text(y_true, y_pred, CLASS_NAMES))

    print("가장 많이 헷갈린 쌍")
    confusions = M.top_confusions(m.confusion, CLASS_NAMES, k=5)
    for c in confusions:
        print(f"  {c['true']:<24s} -> {c['pred']:<24s} {c['count']:3d}장 ({c['rate']:.1%})")

    if args.save:
        out = {
            "run": args.run,
            "split": args.split,
            "checkpoint_epoch": ckpt["epoch"],
            "metrics": m.to_dict(),
            "top_confusions": confusions,
            "class_names": CLASS_NAMES,
            "criteria": {
                "S1_macro_f1": {"threshold": S1_MACRO_F1, "value": m.macro_f1,
                                "passed": m.macro_f1 >= S1_MACRO_F1},
                "S2_min_class_f1": {"threshold": S2_MIN_CLASS_F1, "value": worst_f1,
                                    "worst_class": CLASS_NAMES[worst_i],
                                    "passed": worst_f1 >= S2_MIN_CLASS_F1},
            },
        }
        path = run_dir / f"eval_{args.split}.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        plots.confusion_heatmap(m.confusion, CLASS_NAMES, run_dir / f"confusion_{args.split}.png")
        print(f"\n저장: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
