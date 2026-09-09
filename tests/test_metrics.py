"""평가 지표 검증.

이 프로젝트는 정확도가 아니라 macro F1 을 기준으로 삼는다.
그 차이가 실제로 드러나는지를 테스트로 못박아 둔다.
"""

import numpy as np

from src import metrics as M

NAMES = [f"c{i}" for i in range(8)]


def test_perfect_prediction():
    y = np.arange(8)
    m = M.compute(y, y, 8)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert all(f == 1.0 for f in m.per_class_f1)


def test_macro_f1_penalises_ignored_minority_class():
    """다수 클래스만 맞히면 정확도는 높지만 macro F1 은 낮아야 한다.

    이 데이터는 최다/최소가 2.7배라, 정확도만 보면 성능을 잘못 읽는다.
    """
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 100)  # 소수 클래스를 전부 놓침
    m = M.compute(y_true, y_pred, 2)
    assert m.accuracy == 0.9
    assert m.macro_f1 < 0.5


def test_worst_class_finds_minimum():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 0])  # 클래스 2 를 전부 틀림
    m = M.compute(y_true, y_pred, 3)
    idx, f1 = m.worst_class
    assert idx == 2
    assert f1 == 0.0


def test_confusion_matrix_shape():
    y = np.arange(8)
    m = M.compute(y, y, 8)
    cm = np.array(m.confusion)
    assert cm.shape == (8, 8)
    assert cm.trace() == 8


def test_top_confusions_orders_by_count():
    cm = np.zeros((8, 8), dtype=int)
    np.fill_diagonal(cm, 10)
    cm[1, 6] = 7   # eosinophil -> neutrophil 자리
    cm[6, 1] = 3
    cm[0, 4] = 5
    top = M.top_confusions(cm.tolist(), NAMES, k=3)
    assert [t["count"] for t in top] == [7, 5, 3]
    assert top[0]["true"] == "c1" and top[0]["pred"] == "c6"


def test_top_confusions_ignores_diagonal():
    cm = np.zeros((8, 8), dtype=int)
    np.fill_diagonal(cm, 100)
    assert M.top_confusions(cm.tolist(), NAMES) == []


def test_metrics_serialisable():
    y = np.arange(8)
    d = M.compute(y, y, 8).to_dict()
    assert set(d) == {"accuracy", "macro_f1", "weighted_f1", "per_class_f1", "confusion"}
    assert isinstance(d["per_class_f1"][0], float)
