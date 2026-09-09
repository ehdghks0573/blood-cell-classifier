"""결과 이미지 저장. 학습 없이도 데이터를 눈으로 확인할 수 있어야 한다."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

#: matplotlib 기본 폰트에는 한글이 없어 제목·축 라벨이 네모로 나온다.
_KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"]


def use_korean_font() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    # 한글 폰트는 유니코드 마이너스를 안 갖고 있는 경우가 많다.
    plt.rcParams["axes.unicode_minus"] = False


use_korean_font()


def short_names(names: list[str]) -> list[str]:
    return [n.split("(")[0].strip()[:16] for n in names]


def sample_grid(images: np.ndarray, labels: np.ndarray, class_names: list[str],
                path: Path, per_class: int = 6) -> None:
    """클래스마다 몇 장씩 모아 격자로 저장한다.

    라벨이 제대로 붙어 있는지, 클래스가 눈으로 구분되는지 확인하는 용도다.
    데이터를 안 보고 학습부터 돌리면 나중에 원인을 못 찾는다.
    """
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, per_class, figsize=(per_class * 1.3, n_cls * 1.45))

    for c in range(n_cls):
        idx = np.flatnonzero(labels.reshape(-1) == c)[:per_class]
        for j in range(per_class):
            ax = axes[c, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(idx):
                ax.imshow(images[idx[j]])
            else:
                ax.axis("off")
            if j == 0:
                ax.set_ylabel(short_names(class_names)[c], rotation=0, ha="right",
                              va="center", fontsize=8)

    fig.suptitle("BloodMNIST — 클래스별 샘플", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def class_distribution(counts: dict[str, np.ndarray], class_names: list[str], path: Path) -> None:
    """분할별 클래스 분포 막대그래프."""
    short = short_names(class_names)
    x = np.arange(len(short))
    width = 0.8 / len(counts)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (split, c) in enumerate(counts.items()):
        ax.bar(x + i * width, c, width, label=split)

    ax.set_xticks(x + width * (len(counts) - 1) / 2)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("장수")
    ax.set_title("클래스 분포 — 최다/최소 2.7배")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def confusion_heatmap(confusion, class_names: list[str], path: Path,
                      normalize: bool = True) -> None:
    """혼동행렬 히트맵. 어떤 클래스끼리 헷갈리는지가 이 그림에서 드러난다."""
    cm = np.asarray(confusion, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    short = short_names(class_names)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("예측")
    ax.set_ylabel("정답")
    ax.set_title("혼동행렬" + (" (행 정규화)" if normalize else ""))

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, path)


def training_curves(history: list[dict], path: Path) -> None:
    """손실과 macro F1 곡선. 과적합이 시작되는 지점을 여기서 본다."""
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("에폭")
    ax1.set_ylabel("손실")
    ax1.set_title("손실")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["val_macro_f1"] for h in history], color="tab:green")
    ax2.set_xlabel("에폭")
    ax2.set_ylabel("macro F1")
    ax2.set_title("검증 macro F1")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, path)


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
