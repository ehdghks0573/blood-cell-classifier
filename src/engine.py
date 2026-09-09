"""학습과 추론 루프."""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

#: 진행바는 터미널에서만 그린다. 파일로 리다이렉트하면 갱신 한 번마다 한 줄씩
#: 쌓여 로그가 수만 줄이 된다.
_INTERACTIVE = sys.stdout.isatty()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, epochs: int) -> float:
    model.train()
    total, n = 0.0, 0

    bar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False, ncols=80,
               disable=not _INTERACTIVE)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total += loss.item() * labels.size(0)
        n += labels.size(0)
        bar.set_postfix(loss=f"{total / n:.4f}")

    return total / max(1, n)


@torch.no_grad()
def predict(model, loader, device, criterion: nn.Module | None = None):
    """(정답, 예측, 확률, 평균손실) 을 돌려준다."""
    model.eval()
    trues, preds, probs = [], [], []
    total, n = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            total += criterion(logits, labels).item() * labels.size(0)
            n += labels.size(0)

        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(labels.cpu().numpy())

    return (
        np.concatenate(trues),
        np.concatenate(preds),
        np.concatenate(probs),
        total / n if n else 0.0,
    )


def class_weights(counts: np.ndarray, device) -> torch.Tensor:
    """빈도의 역수를 정규화한 가중치.

    클래스 불균형이 2.7배로 심하지는 않아서 없어도 될 수 있다.
    있고 없고를 비교해 실험 표에 남기는 것이 이 프로젝트의 목적에 맞다.
    """
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)
