"""Grad-CAM 히트맵 생성.

    python explain.py --run baseline_resnet18_28px

두 가지를 만든다.
  1) 클래스별 대표 히트맵 — 모델이 각 세포에서 어디를 보는가 (S4)
  2) 정답/오답 비교 — 틀렸을 때는 어디를 보고 있었는가 (S5 의 근거)

두 번째가 이 프로젝트의 핵심이다. 모델이 세포가 아니라 배경이나 이미지
가장자리를 보고 판단하고 있다면, 정확도 숫자만으로는 절대 알 수 없다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    BloodDataset,
    build_transform,
    denormalize,
    load_split,
)
from src.cli import guard, load_run, pick_device  # noqa: E402
from src.gradcam import GradCAM, overlay  # noqa: E402
from src.model import target_layer  # noqa: E402
from src.plots import short_names, use_korean_font  # noqa: E402

use_korean_font()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM 히트맵 생성")
    p.add_argument("--run", required=True, help="runs/ 아래 실행 이름")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--per-class", type=int, default=4, help="클래스당 표시할 장수")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def load_checkpoint(run_dir: Path, device):
    """`src.cli.load_run` 으로 옮겼다. 예전 호출부를 위해 남겨 둔다."""
    return load_run(run_dir.name, ROOT, device)


def class_gallery(model, layer, dataset, device, out: Path, per_class: int) -> None:
    """클래스마다 몇 장씩 골라 원본과 히트맵을 나란히 보여준다."""
    labels = dataset.labels
    rows, cols = NUM_CLASSES, per_class * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.5))

    with GradCAM(model, layer) as cam:
        for c in range(NUM_CLASSES):
            idx = np.flatnonzero(labels == c)[:per_class]
            for j in range(per_class):
                ax_img, ax_cam = axes[c, j * 2], axes[c, j * 2 + 1]
                for ax in (ax_img, ax_cam):
                    ax.set_xticks([])
                    ax.set_yticks([])

                if j >= len(idx):
                    ax_img.axis("off")
                    ax_cam.axis("off")
                    continue

                image, _ = dataset[int(idx[j])]
                x = image.unsqueeze(0).to(device)
                heat, pred, probs = cam(x, class_index=c)

                base = denormalize(image).cpu().numpy()
                ax_img.imshow(np.transpose(base, (1, 2, 0)))
                ax_cam.imshow(overlay(base, heat))
                ax_cam.set_title(f"{probs[c]:.2f}", fontsize=6, pad=1,
                                 color="tab:green" if pred == c else "tab:red")

                if j == 0:
                    ax_img.set_ylabel(short_names(CLASS_NAMES)[c], rotation=0, ha="right",
                                      va="center", fontsize=8)

    fig.suptitle("클래스별 Grad-CAM — 각 쌍은 (원본, 히트맵), 숫자는 해당 클래스 확률",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out)


def hit_vs_miss(model, layer, dataset, device, out: Path, n: int = 6) -> None:
    """맞힌 경우와 틀린 경우의 히트맵을 나란히 비교한다.

    틀린 쪽에서 모델이 세포 바깥을 보고 있다면, 그것이 오분류의 원인이다.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            preds.append(model(images.to(device)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    wrong = np.flatnonzero(preds != dataset.labels)[:n]
    right = np.flatnonzero(preds == dataset.labels)[:n]
    if len(wrong) == 0:
        print("  오분류가 없어 비교 그림을 건너뜁니다.")
        return

    fig, axes = plt.subplots(2, len(wrong) + len(right), figsize=((len(wrong) + len(right)) * 1.6, 4))
    picks = [(i, True) for i in right[:len(wrong)]] + [(i, False) for i in wrong]

    with GradCAM(model, layer) as cam:
        for col, (i, correct) in enumerate(picks):
            image, true_c = dataset[int(i)]
            x = image.unsqueeze(0).to(device)
            heat, pred, probs = cam(x)

            base = denormalize(image).cpu().numpy()
            axes[0, col].imshow(np.transpose(base, (1, 2, 0)))
            axes[1, col].imshow(overlay(base, heat))
            for ax in (axes[0, col], axes[1, col]):
                ax.set_xticks([])
                ax.set_yticks([])

            short = short_names(CLASS_NAMES)
            title = short[true_c] if correct else f"{short[true_c]}\n→{short[pred]}"
            axes[0, col].set_title(title, fontsize=7,
                                   color="tab:green" if correct else "tab:red")

    axes[0, 0].set_ylabel("원본", rotation=0, ha="right", va="center", fontsize=9)
    axes[1, 0].set_ylabel("히트맵", rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("맞힌 경우(초록) vs 틀린 경우(빨강) — 틀릴 때 모델은 어디를 보는가", fontsize=11)
    fig.tight_layout()
    _save(fig, out)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = ROOT / "runs" / args.run

    device = pick_device(args.cpu)
    model, ckpt = load_run(args.run, ROOT, device)
    layer = target_layer(model, ckpt["arch"])

    imgs, lbls = load_split(args.data_root, args.size, args.split)
    dataset = BloodDataset(imgs, lbls, build_transform(ckpt["input_size"], train=False))

    print(f"실행 {args.run} · {ckpt['arch']} · 에폭 {ckpt['epoch']} "
          f"(val macro F1 {ckpt['val_macro_f1']:.4f})")
    print(f"{args.split} {len(dataset):,}장으로 히트맵 생성")

    class_gallery(model, layer, dataset, device, run_dir / "gradcam_classes.png", args.per_class)
    hit_vs_miss(model, layer, dataset, device, run_dir / "gradcam_hit_vs_miss.png")
    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
