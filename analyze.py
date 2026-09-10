"""라벨 모호성 가설을 검증한다 (성공 기준 S5).

    python analyze.py --run baseline_resnet18_28px --save

`docs/results.md` 4절에서 "남은 오분류는 모델의 결함이 아니라 라벨이 연속선상에
있어서 모호한 탓"이라는 가설을 세웠다. 여기서는 그 가설이 **틀렸다면 어떤 숫자가
나와야 하는지**를 먼저 정해 두고(`src/ambiguity.py` 참고), 실제 확률 출력으로
확인한다. 가설이 깨지면 깨진 대로 적는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import ambiguity as A  # noqa: E402
from src import plots  # noqa: E402
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
    p = argparse.ArgumentParser(description="라벨 모호성 가설 검증 (S5)")
    p.add_argument("--run", required=True, help="runs/ 아래 실행 이름")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save", action="store_true",
                   help="ambiguity.json 과 확신도 그래프를 runs/<run>/ 에 저장한다")
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

    print(f"실행     : {args.run} · {ckpt['arch']} · 에폭 {ckpt['epoch']}")
    print(f"평가 대상: {args.split} {len(dataset):,}장")
    print(f"삼각지대 : {', '.join(CLASS_NAMES[i] for i in A.TRIANGLE)}")
    print()

    y_true, y_pred, probs, _ = predict(model, loader, device)
    result = A.analyze(y_true, probs, CLASS_NAMES)

    print("확신도 비교")
    print(f"  {'묶음':<14s} {'장수':>6s} {'확신도':>8s} {'정답확률':>9s} "
          f"{'마진':>8s} {'엔트로피':>9s} {'정답 2순위':>11s}")
    for g in result.groups:
        print(f"  {g.name:<14s} {g.n:6d} {g.mean_confidence:8.3f} {g.mean_true_prob:9.3f} "
              f"{g.mean_margin:8.3f} {g.mean_entropy:9.3f} {g.true_in_top2:10.1%}")
    print()

    print("혼동 쌍별 확신도")
    for r in result.pairs:
        mark = "△" if r["both_in_triangle"] else " "
        print(f"  {mark} {r['true'][:22]:<24s} -> {r['pred'][:22]:<24s} "
              f"{r['count']:3d}장  확신 {r['mean_confidence']:.3f}  "
              f"정답확률 {r['mean_true_prob']:.3f}")
    print()

    print("맞힌 예측의 2순위  (정답 수천 장으로 재므로 오분류보다 표본이 100배 크다)")
    print("  ※ '어느 클래스가 2순위인가'는 여덟 클래스 전부 쏠려 나온다 — 구분에 못 쓴다.")
    print("     실제 신호는 마지막 열, 2순위에 **실린 확률**이다.")
    for r in result.runner_ups:
        mark = "△" if r["both_in_triangle"] else " "
        print(f"  {mark} {r['class'][:22]:<24s} 2순위 {r['runner_up'][:22]:<24s} "
              f"{r['share']:6.1%}  n={r['n_correct']:<5d} "
              f"2순위 확률 {r['runner_up_prob']:.4f}")
    print()

    print("가설의 예측  (95% 부트스트랩 구간이 0 을 걸치면 판정보류)")
    for name, c in result.verdict["checks"].items():
        mark = {True: "맞음", False: "빗나감", None: "판정보류"}[c["passed"]]
        print(f"  [{mark:<4s}] {name:<18s} {c['evidence']}")
    print()
    v = result.verdict
    print(f"판정: 맞음 {v['passed']} · 빗나감 {v['failed']} · 판정보류 {v['undecided']}"
          f"  (총 {v['total']})")
    print(f"      {v['summary']}")

    if args.save:
        out = {
            "run": args.run,
            "split": args.split,
            "checkpoint_epoch": ckpt["epoch"],
            "triangle": [CLASS_NAMES[i] for i in A.TRIANGLE],
            **result.to_dict(),
        }
        path = run_dir / "ambiguity.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        _confidence_plot(y_true, y_pred, probs, run_dir / "confidence.png")
        print(f"\n저장: {path}")
        print(f"저장: {run_dir / 'confidence.png'}")

    return 0


def _confidence_plot(y_true, y_pred, probs, path: Path) -> None:
    """세 묶음의 확신도 분포. 표보다 이 그림이 차이를 빨리 보여준다."""
    import matplotlib.pyplot as plt
    import numpy as np

    conf = probs.max(axis=1)
    wrong = y_pred != y_true
    tri = A.in_triangle(y_true) & A.in_triangle(y_pred)

    groups = [
        ("정답", conf[~wrong], "tab:green"),
        ("삼각지대 오류", conf[wrong & tri], "tab:orange"),
        ("그 밖의 오류", conf[wrong & ~tri], "tab:red"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    bins = np.linspace(0.125, 1.0, 25)
    for ax, (name, values, color) in zip(axes, groups):
        ax.hist(values, bins=bins, color=color, alpha=0.85)
        mean = values.mean() if values.size else float("nan")
        ax.axvline(mean, color="black", ls="--", lw=1)
        ax.set_title(f"{name} · {values.size}장 · 평균 {mean:.3f}", fontsize=10)
        ax.set_xlabel("예측 클래스에 준 확률")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("장수")

    fig.suptitle("모델은 언제 확신을 잃는가", fontsize=12)
    fig.tight_layout()
    plots._save(fig, path)


if __name__ == "__main__":
    sys.exit(main())
