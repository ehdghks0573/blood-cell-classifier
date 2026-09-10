"""오분류를 눈으로 판단하기 위한 비교 격자 (Day 10, S5).

    python inspect_errors.py --run res112_resnet18

`analyze.py` 로 확률 분포를 재 봤지만, 남은 경쟁이 **라벨이 모호해서**인지
**모델이 못 잡아서**인지는 숫자로 가릴 수 없었다. 오분류가 70~80장뿐이라
표본이 부족하고, 그건 모델을 더 잘 만들어도 해결되지 않는다.

남은 방법은 사람이 보는 것이다. 그런데 **틀린 이미지만 늘어놓으면 판단할 수
없다.** "이게 미성숙 과립구인가 호중구인가"는 그 두 클래스의 전형이 어떻게
생겼는지 옆에 있어야 물을 수 있는 질문이다. 그래서 한 줄에

    [틀린 이미지] [히트맵] | [정답 클래스 대표 3장] | [예측 클래스 대표 3장]

을 나란히 놓는다. 보는 사람이 답해야 할 질문은 하나다 —
**이 이미지는 왼쪽 무리와 오른쪽 무리 중 어느 쪽에 더 가까운가.**

왼쪽(정답)에 가까워 보이면 모델이 틀린 것이고, 오른쪽(예측)에 가까워 보이면
**라벨이 의심스러운 것**이다. 어느 쪽도 아니면 그 이미지는 원래 애매한 것이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cli import guard, load_run, pick_device  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_KO,
    BloodDataset,
    build_transform,
    denormalize,
    load_split,
)
from src.gradcam import GradCAM, overlay  # noqa: E402
from src.model import target_layer  # noqa: E402
from src.plots import use_korean_font  # noqa: E402

use_korean_font()

#: 미성숙 과립구. `analyze.py` 결과에서 여러 클래스와 경쟁하는 것으로 나온 클래스다.
DEFAULT_FOCUS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="오분류 비교 격자 (S5)")
    p.add_argument("--run", required=True, help="runs/ 아래 실행 이름")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--focus", type=int, default=DEFAULT_FOCUS,
                   help=f"이 클래스가 낀 오분류만 본다 (기본 {DEFAULT_FOCUS} "
                        f"= {CLASS_NAMES[DEFAULT_FOCUS]})")
    p.add_argument("--all-classes", action="store_true",
                   help="특정 클래스로 좁히지 않고 전체 혼동 쌍을 본다")
    p.add_argument("--from-json",
                   help="tools/consensus_errors.py 가 저장한 JSON. 거기 적힌 장만 그린다 "
                        "— 여러 모델이 공통으로 틀린 장이라 가장 할 말이 많다")
    p.add_argument("--pairs", type=int, default=4, help="상위 몇 개 혼동 쌍을 그릴지")
    p.add_argument("--per-pair", type=int, default=6, help="쌍마다 몇 장을 볼지")
    p.add_argument("--refs", type=int, default=3, help="대표 이미지 장수")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def prototypes(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
               cls: int, n: int) -> np.ndarray:
    """그 클래스의 **범위** — 확신도 전 구간에서 고르게 뽑은 맞힌 장들.

    처음에는 "가장 확신 있게 맞힌" 장만 뽑았는데, 그러면 비교가 왜곡된다.
    미성숙 과립구는 myelocyte·metamyelocyte·promyelocyte 가 한 라벨에 묶인
    **이질적인 클래스**다. 전형만 보여 주면 그 클래스의 폭이 실제보다 좁아
    보이고, 틀린 이미지가 실제보다 더 "라벨이 잘못된 것처럼" 보인다.

    확신 높은 쪽부터 낮은 쪽까지 고르게 뽑으면, 그 클래스가 어디까지를
    포함하는지 — 특히 경계에 있는 장이 어떻게 생겼는지 — 함께 볼 수 있다.
    """
    ok = np.flatnonzero((y_true == cls) & (y_pred == cls))
    if ok.size == 0:
        return np.zeros(0, dtype=int)

    ordered = ok[np.argsort(-probs[ok, cls])]
    if ordered.size <= n:
        return ordered
    # 0%(가장 확신) ~ 100%(경계) 구간을 n 등분해서 집는다
    at = np.linspace(0, ordered.size - 1, n).round().astype(int)
    return ordered[at]


def pair_figure(images: np.ndarray, dataset, model, layer, device,
                probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                true_c: int, pred_c: int, picks: np.ndarray,
                refs: int, out: Path, total: int) -> None:
    """혼동 쌍 하나에 대한 비교 격자."""
    ref_true = prototypes(probs, y_true, y_pred, true_c, refs)
    ref_pred = prototypes(probs, y_true, y_pred, pred_c, refs)

    n_rows = len(picks)
    n_cols = 2 + refs * 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.25, n_rows * 1.45),
                             squeeze=False)

    with GradCAM(model, layer) as cam:
        for r, i in enumerate(picks):
            image, _ = dataset[int(i)]
            heat, _, _ = cam(image.unsqueeze(0).to(device), class_index=pred_c)
            base = denormalize(image).cpu().numpy()

            axes[r][0].imshow(images[i], interpolation="nearest")
            axes[r][1].imshow(overlay(base, heat))

            # 왼쪽엔 정답 클래스의 전형, 오른쪽엔 예측 클래스의 전형
            for k in range(refs):
                axes[r][2 + k].imshow(
                    images[ref_true[k]] if k < len(ref_true) else np.zeros_like(images[i]),
                    interpolation="nearest")
                axes[r][2 + refs + k].imshow(
                    images[ref_pred[k]] if k < len(ref_pred) else np.zeros_like(images[i]),
                    interpolation="nearest")

            axes[r][0].set_ylabel(f"확신 {probs[i, pred_c]:.2f}\n정답 {probs[i, true_c]:.2f}",
                                  rotation=0, ha="right", va="center", fontsize=7)

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            # 비교할 두 무리 사이에 경계를 준다
            for side in ("left", "right", "top", "bottom"):
                axes[r][c].spines[side].set_visible(False)
        # 히트맵 오른쪽(=비교 대상과의 경계)과 정답 무리 오른쪽(=두 무리 사이)
        for boundary in (1, 1 + refs):
            axes[r][boundary].spines["right"].set_visible(True)
            axes[r][boundary].spines["right"].set_linewidth(2)

    headers = (["틀린 이미지", "히트맵"]
               + [f"정답: {CLASS_NAMES_KO[true_c]}  (전형 → 경계)"] + [""] * (refs - 1)
               + [f"예측: {CLASS_NAMES_KO[pred_c]}  (전형 → 경계)"] + [""] * (refs - 1))
    for c, text in enumerate(headers):
        if text:
            axes[0][c].set_title(text, fontsize=8, pad=6, loc="left")

    fig.suptitle(
        f"{CLASS_NAMES[true_c]} → {CLASS_NAMES[pred_c]}  "
        f"(전체 {total}장 중 확신 높은 {len(picks)}장)\n"
        f"물어볼 것: 왼쪽 이미지는 가운데 무리와 오른쪽 무리 중 어느 쪽에 가까운가\n"
        f"대표는 각 클래스를 맞힌 장에서 확신 높은 쪽부터 경계까지 고르게 뽑았다",
        fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def contact_sheet(images: np.ndarray, cases: list[dict], out: Path,
                  cols: int = 7) -> None:
    """공통 오분류 **전부**를 한 장에 늘어놓는다.

    쌍별 격자는 근거를 따지기에는 좋지만 "그래서 몇 장인데?"에 답하지 못한다.
    26장을 한 화면에 놓으면 크기가 손에 잡힌다 — 3,421장 중 이것뿐이라는 것도,
    그런데도 우연으로는 설명되지 않는다는 것도.
    """
    n = len(cases)
    rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.95),
                             squeeze=False)

    short = {name: ko for name, ko in zip(CLASS_NAMES, CLASS_NAMES_KO)}

    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            i = r * cols + c
            if i >= n:
                ax.axis("off")
                continue

            case = cases[i]
            ax.imshow(images[case["index"]], interpolation="nearest")
            conf = case.get("min_confidence", 0.0)
            ax.set_title(
                f"{short.get(case['true'], case['true'])[:6]}"
                f" → {short.get(case['pred'], case['pred'])[:6]}\n"
                f"#{case['index']} · {conf:.2f}",
                fontsize=6.5, pad=2,
                # 전부가 확신에 차서 틀린 것은 아니다. 그 차이를 색으로 남긴다.
                color="tab:red" if conf >= 0.9 else "0.35")
            for side in ax.spines.values():
                side.set_color("tab:red" if conf >= 0.9 else "0.75")
                side.set_linewidth(1.4 if conf >= 0.9 else 0.8)

    confident = sum(1 for c in cases if c.get("min_confidence", 0) >= 0.9)
    fig.suptitle(
        f"모든 모델이 같은 답으로 틀린 {n}장 — 빨강은 전부 90% 이상 확신한 {confident}장",
        fontsize=11)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = ROOT / "runs" / args.run

    device = pick_device(args.cpu)
    model, ckpt = load_run(args.run, ROOT, device)
    layer = target_layer(model, ckpt["arch"])

    images, lbls = load_split(args.data_root, args.size, args.split)
    dataset = BloodDataset(images, lbls, build_transform(ckpt["input_size"], train=False))

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    from src.engine import predict
    y_true, y_pred, probs, _ = predict(model, loader, device)

    wrong = np.flatnonzero(y_pred != y_true)
    if args.from_json:
        # 여러 모델이 공통으로 틀린 장만 남긴다. 이 모델 하나가 틀린 것과
        # 모든 모델이 틀린 것은 무게가 다르다.
        consensus = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        keep_idx = {c["index"] for c in consensus["cases"]}
        wrong = np.array([i for i in wrong if int(i) in keep_idx], dtype=int)
        title = f"{len(consensus['runs'])}개 모델 공통"

        # 전체를 한 장에. 쌍별 격자는 근거를 따지는 용도이고, 이건 규모를 보는 용도다.
        sheet = ROOT / "runs" / args.run / "errors" / "consensus_all.png"
        contact_sheet(images, consensus["cases"], sheet)
    elif args.all_classes:
        title = "전체"
    else:
        keep = (y_true[wrong] == args.focus) | (y_pred[wrong] == args.focus)
        wrong = wrong[keep]
        title = CLASS_NAMES[args.focus]

    print(f"실행 {args.run} · {ckpt['arch']} · 에폭 {ckpt['epoch']}")
    print(f"{args.split} {len(dataset):,}장 중 오분류 {int((y_pred != y_true).sum())}장, "
          f"그중 {title} 관련 {len(wrong)}장")

    if len(wrong) == 0:
        print("볼 것이 없습니다.")
        return 0

    # 혼동 쌍별로 묶고, 장수가 많은 쌍부터
    groups: dict[tuple[int, int], list[int]] = {}
    for i in wrong:
        groups.setdefault((int(y_true[i]), int(y_pred[i])), []).append(int(i))
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:args.pairs]

    out_dir = run_dir / "errors"
    manifest = []
    print()
    for (true_c, pred_c), idx in ordered:
        # 확신이 높은 것부터. 확신에 차서 틀린 장이 가장 할 말이 많다 —
        # 라벨이 틀렸거나, 모델이 크게 잘못 배웠거나 둘 중 하나다.
        idx = np.array(idx)
        picks = idx[np.argsort(-probs[idx, pred_c])][:args.per_pair]

        name = f"{CLASS_NAMES[true_c].replace(' ', '_')}__to__{CLASS_NAMES[pred_c].replace(' ', '_')}.png"
        print(f"{CLASS_NAMES[true_c]} → {CLASS_NAMES[pred_c]}  {len(idx)}장 "
              f"(확신 평균 {probs[idx, pred_c].mean():.3f})")
        pair_figure(images, dataset, model, layer, device, probs, y_true, y_pred,
                    true_c, pred_c, picks, args.refs, out_dir / name, total=len(idx))

        manifest.append({
            "true": CLASS_NAMES[true_c],
            "pred": CLASS_NAMES[pred_c],
            "count": len(idx),
            "mean_confidence": round(float(probs[idx, pred_c].mean()), 4),
            "shown": [int(i) for i in picks],
            "confidence_shown": [round(float(probs[i, pred_c]), 4) for i in picks],
            "true_prob_shown": [round(float(probs[i, true_c]), 4) for i in picks],
            "file": name,
        })

    path = out_dir / "manifest.json"
    path.write_text(json.dumps(
        {"run": args.run, "split": args.split, "focus": title, "pairs": manifest},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n색인: {path}")
    print("\n보는 법 — 각 줄에서 왼쪽 이미지가 가운데(정답) 무리와 오른쪽(예측) 무리 중")
    print("어느 쪽에 가까운지 봅니다. 오른쪽에 가까우면 라벨을 의심할 근거가 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
