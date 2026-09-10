"""오분류 비교 격자에서 '대표 이미지'를 고르는 규칙.

이 함수가 조용히 틀리면 **그림은 멀쩡히 그려지고 결론만 잘못된다.** 대표를
전형에서만 뽑으면 클래스의 폭이 좁아 보이고, 틀린 이미지가 실제보다 더
"라벨이 잘못된 것처럼" 보인다. 그래서 규칙을 테스트로 못박는다.
"""

import numpy as np

from inspect_errors import prototypes

N_CLASSES = 8


def _probs(confidences: dict[int, float], cls: int, n_classes: int = N_CLASSES):
    """인덱스별 정답 확률을 지정해 확률 배열을 만든다."""
    p = np.full((max(confidences) + 1, n_classes), 0.01)
    for i, conf in confidences.items():
        p[i] = (1 - conf) / (n_classes - 1)
        p[i, cls] = conf
    return p


def test_returns_empty_when_class_never_predicted_correctly():
    y_true = np.array([3, 3])
    y_pred = np.array([5, 5])  # 한 장도 못 맞힘
    probs = _probs({0: 0.9, 1: 0.8}, cls=3)
    assert prototypes(probs, y_true, y_pred, 3, n=4).size == 0


def test_ignores_wrong_predictions():
    """맞힌 장만 대표가 될 수 있다. 틀린 장은 그 클래스의 전형이 아니다."""
    y_true = np.array([3, 3, 3])
    y_pred = np.array([3, 5, 3])  # 가운데는 틀림
    probs = _probs({0: 0.9, 1: 0.99, 2: 0.7}, cls=3)
    assert set(prototypes(probs, y_true, y_pred, 3, n=3)) == {0, 2}


def test_returns_all_when_fewer_than_requested():
    y_true = np.array([3, 3])
    y_pred = np.array([3, 3])
    probs = _probs({0: 0.9, 1: 0.8}, cls=3)
    assert prototypes(probs, y_true, y_pred, 3, n=5).size == 2


def test_spans_confidence_range_instead_of_taking_only_the_top():
    """이 테스트가 이 파일의 핵심이다.

    확신도 0.99 부터 0.30 까지 20장이 있을 때, 상위 4장(전부 0.9 이상)만
    뽑으면 안 된다. 가장 확신한 장과 경계에 있는 장이 모두 들어와야 한다.
    """
    n = 20
    confs = np.linspace(0.99, 0.30, n)
    y_true = np.full(n, 3)
    y_pred = np.full(n, 3)
    probs = _probs({i: float(c) for i, c in enumerate(confs)}, cls=3)

    picked = prototypes(probs, y_true, y_pred, 3, n=4)
    conf_picked = probs[picked, 3]

    assert len(picked) == 4
    assert conf_picked[0] == confs.max()   # 전형
    assert conf_picked[-1] == confs.min()  # 경계
    assert list(conf_picked) == sorted(conf_picked, reverse=True)
    # 상위 4장만 뽑는 옛 방식이면 최솟값이 0.9 를 넘는다
    assert conf_picked.min() < 0.5


def test_order_is_confident_first():
    n = 10
    y_true = np.full(n, 6)
    y_pred = np.full(n, 6)
    probs = _probs({i: 0.5 + i * 0.04 for i in range(n)}, cls=6)

    picked = prototypes(probs, y_true, y_pred, 6, n=3)
    assert picked[0] == n - 1  # 확신이 가장 높은 인덱스
