"""평가 지표.

정확도가 아니라 macro F1 을 기준으로 삼는다. 이 데이터는 최다/최소 클래스가
2.7배 차이라, 상위 세 클래스(neutrophil, eosinophil, immature granulocytes)만
잘 맞혀도 전체의 54% 라 정확도가 높게 나온다. 그러면 성능을 잘못 읽게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: list[float] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)

    @property
    def worst_class(self) -> tuple[int, float]:
        """F1 이 가장 낮은 클래스. 성공 기준 S2 판정에 쓴다."""
        i = int(np.argmin(self.per_class_f1))
        return i, self.per_class_f1[i]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_f1": self.per_class_f1,
            "confusion": self.confusion,
        }


def compute(y_true, y_pred, num_classes: int = 8) -> Metrics:
    labels = list(range(num_classes))
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        per_class_f1=[float(x) for x in f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)],
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def report_text(y_true, y_pred, class_names: list[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=class_names,
        labels=list(range(len(class_names))), digits=3, zero_division=0,
    )


def top_confusions(confusion: list[list[int]], class_names: list[str], k: int = 3):
    """가장 많이 헷갈린 (정답, 예측) 쌍 상위 k개.

    오류 분석(S5)의 출발점이다. 어떤 세포끼리 헷갈리는지 알아야
    무엇을 개선할지 정할 수 있다.
    """
    cm = np.asarray(confusion)
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                total = cm[i].sum()
                pairs.append({
                    "true": class_names[i],
                    "pred": class_names[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / total) if total else 0.0,
                })
    pairs.sort(key=lambda p: -p["count"])
    return pairs[:k]
