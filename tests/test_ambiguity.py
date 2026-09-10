"""라벨 모호성 검증 지표의 단위 테스트.

이 코드가 하는 일은 "가설이 맞았는지 틀렸는지 판정"이다. 판정 코드가 조용히
잘못되면 **결론이 통째로 뒤집히는데 아무도 모른다.** 그래서 정답이 뻔한
인공 데이터로, 맞는 경우와 틀리는 경우를 둘 다 못박아 둔다.
"""

import numpy as np
import pytest

from src import ambiguity as A

NAMES = [f"c{i}" for i in range(8)]
IG, MONO, NEU = A.TRIANGLE  # 3, 5, 6


def onehot(cls: int, p: float = 1.0, other: int | None = None) -> np.ndarray:
    """`cls` 에 p, 나머지 하나(`other`)에 1-p 를 준 확률 분포."""
    row = np.zeros(8)
    row[cls] = p
    if other is not None:
        row[other] = 1.0 - p
    elif p < 1.0:
        row[(cls + 1) % 8] = 1.0 - p
    return row


def test_entropy_is_zero_when_certain():
    assert A.entropy(onehot(0)[None, :])[0] == 0.0


def test_entropy_is_maximal_when_uniform():
    uniform = np.full((1, 8), 1 / 8)
    assert A.entropy(uniform)[0] == np.log(8)


def test_in_triangle_selects_three_classes():
    labels = np.arange(8)
    assert A.in_triangle(labels).tolist() == [
        False, False, False, True, False, True, True, False]


def test_group_stats_on_empty_mask_is_not_a_crash():
    y = np.array([0, 1])
    probs = np.stack([onehot(0), onehot(1)])
    g = A.group_stats("빈 묶음", y, probs, np.zeros(2, dtype=bool))
    assert g.n == 0
    assert np.isnan(g.mean_confidence)


def test_margin_is_zero_on_correct_predictions():
    y = np.array([0, 1])
    probs = np.stack([onehot(0, 0.8), onehot(1, 0.7)])
    g = A.group_stats("정답", y, probs, np.ones(2, dtype=bool))
    assert g.mean_margin == 0.0
    assert g.true_in_top2 == 1.0


def test_confident_wrong_counts_only_high_probability_errors():
    y = np.array([0, 0])
    probs = np.stack([
        onehot(1, 0.95, other=0),  # 확신에 차서 틀림
        onehot(1, 0.55, other=0),  # 헷갈리며 틀림
    ])
    g = A.group_stats("오류", y, probs, np.ones(2, dtype=bool))
    assert g.confident_wrong == 1


def dist(mass: dict[int, float]) -> np.ndarray:
    row = np.zeros(8)
    for cls, p in mass.items():
        row[cls] = p
    return row


def error_row(pred: int, true: int, decoy: int, conf: float, ambiguous: bool) -> np.ndarray:
    """틀린 예측 한 장의 확률 분포.

    `ambiguous=True` 면 남은 확률이 정답으로 가서 정답이 2순위에 남고,
    False 면 엉뚱한 클래스로 가서 정답이 2순위에도 못 든다. 확신도와 2순위
    여부는 서로 독립된 신호이므로 따로 정해 줘야 판정 코드를 제대로 시험한다.
    """
    rest = 1.0 - conf
    second, third = (true, decoy) if ambiguous else (decoy, true)
    return dist({pred: conf, second: rest * 0.9, third: rest * 0.1})


def _dataset(tri_conf: float, other_conf: float, tri_ambiguous: bool = True,
             other_ambiguous: bool = False, n: int = 10):
    """삼각지대 오류와 그 밖의 오류를 같은 장수로 만든 인공 데이터."""
    y, p = [], []
    for _ in range(n):
        y.append(NEU)
        p.append(error_row(IG, NEU, MONO, tri_conf, tri_ambiguous))
    for _ in range(n):
        y.append(0)
        p.append(error_row(1, 0, 2, other_conf, other_ambiguous))
    for _ in range(n):
        y.append(7)
        p.append(onehot(7, 0.99))  # 정답
    return np.array(y), np.stack(p)


def test_hypothesis_holds_when_triangle_errors_are_uncertain():
    """삼각지대에서 덜 확신하면 세 예측이 모두 맞아야 한다."""
    y, probs = _dataset(tri_conf=0.55, other_conf=0.95)
    r = A.analyze(y, probs, NAMES)

    # P4 는 삼각지대를 **맞힌** 장으로 재는데 이 인공 데이터에는 없다.
    # 오분류로 재는 P1~P3 만 본다.
    error_checks = {k: v for k, v in r.verdict["checks"].items() if not k.startswith("P4")}

    assert r.verdict["comparable"] is True
    assert all(c["passed"] is True for c in error_checks.values())
    assert r.verdict["checks"]["P4_2순위에_실린_확률"]["passed"] is None


def test_hypothesis_breaks_when_triangle_errors_are_confident():
    """확신에 차서 틀리면 가설은 깨져야 한다. 깨지는 쪽도 반드시 검출돼야 한다."""
    y, probs = _dataset(tri_conf=0.97, other_conf=0.55,
                        tri_ambiguous=False, other_ambiguous=True)
    r = A.analyze(y, probs, NAMES)

    assert r.verdict["passed"] == 0
    assert "모호성이 아니라" in r.verdict["summary"]


def test_noise_sized_difference_is_not_declared_a_result():
    """이 테스트가 이 파일에서 가장 중요하다.

    실제로 겪은 일이다. 부등호로만 비교하던 판정 코드가, 같은 데이터로 학습한
    두 모델에 대해 `0.818 vs 0.768` 과 `0.790 vs 0.794` 를 보고 각각 3개 중
    1개 / 3개 모두 맞음이라는 **정반대 판정**을 냈다. 40장과 30장짜리 평균의
    차이는 그 정도로 쉽게 흔들린다. 부트스트랩 구간이 0 을 걸치면 판정을
    보류해야 한다.
    """
    # 두 묶음의 확신도를 폭 0.5 로 넓게 퍼뜨리고, 평균만 정확히 0.01 어긋나게
    # 만든다. 난수를 쓰면 시드에 따라 판정이 바뀌므로 값을 직접 정한다.
    spread = np.linspace(0.45, 0.95, 40)
    y, p = [], []
    for conf in spread:
        y.append(NEU)
        p.append(error_row(IG, NEU, MONO, float(conf), ambiguous=True))
    for conf in spread + 0.01:
        y.append(0)
        p.append(error_row(1, 0, 2, float(conf), ambiguous=True))
    y, probs = np.array(y), np.stack(p)

    r = A.analyze(y, probs, NAMES)
    check = r.verdict["checks"]["P1_덜_확신한다"]

    assert r.verdict["comparable"] is True
    assert check["diff"] == pytest.approx(-0.01, abs=1e-6)  # 방향은 가설과 맞지만
    assert check["ci"][0] < 0 < check["ci"][1]  # 구간이 0 을 걸친다
    assert check["passed"] is None  # 따라서 결과로 삼지 않는다


def test_bootstrap_interval_is_a_point_when_values_are_constant():
    a = np.full(50, 0.8)
    b = np.full(50, 0.5)
    lo, hi = A._bootstrap_ci(a, b)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.8, 0.1, 40), rng.normal(0.7, 0.1, 30)
    assert A._bootstrap_ci(a, b) == A._bootstrap_ci(a, b)


def test_verdict_is_withheld_when_samples_are_too_few():
    y, probs = _dataset(tri_conf=0.55, other_conf=0.95, n=2)
    r = A.analyze(y, probs, NAMES)

    assert r.verdict["comparable"] is False
    assert all(c["passed"] is None for c in r.verdict["checks"].values())


def test_error_outside_triangle_is_not_counted_as_triangle():
    """정답만 삼각지대이고 예측이 밖으로 나간 오류는 '그 밖'으로 세야 한다."""
    y = np.array([NEU] * 6)
    probs = np.stack([onehot(0, 0.9, other=NEU)] * 6)  # neutrophil -> c0
    r = A.analyze(y, probs, NAMES)

    tri = next(g for g in r.groups if g.name == "삼각지대 오류")
    other = next(g for g in r.groups if g.name == "그 밖의 오류")
    assert tri.n == 0
    assert other.n == 6


def _third(cls: int, second: int) -> int:
    """`cls` 와도 `second` 와도 겹치지 않는 클래스 하나.

    겹치면 `dist` 의 딕셔너리 키가 덮어써져 예측 자체가 바뀐다.
    """
    return next(c for c in range(8) if c not in (cls, second))


def _correct_rows(cls: int, second: int, n: int) -> tuple[list[int], list[np.ndarray]]:
    """`cls` 를 맞혔고 2순위가 `second` 인 장 n 개."""
    row = dist({cls: 0.9, second: 0.07, _third(cls, second): 0.03})
    return [cls] * n, [row] * n


def test_runner_up_detects_neighbour_bias_on_correct_predictions():
    """맞힌 장의 2순위가 이웃 클래스로 쏠리면 우연 초과로 잡혀야 한다."""
    y, p = [], []
    for cls, second in ((NEU, IG), (IG, NEU), (MONO, IG)):
        ys, ps = _correct_rows(cls, second, 200)
        y += ys
        p += ps
    rows = A.runner_up_rows(np.array(y), np.stack(p), NAMES)

    assert len(rows) == 3
    for r in rows:
        assert r["share"] == 1.0
        assert r["above_chance"] is True
        assert r["both_in_triangle"] is True
        assert r["ci"][0] > r["chance"]


def test_runner_up_does_not_cry_wolf_on_chance_level_second_place():
    """2순위가 고르게 흩어져 있으면 우연 수준으로 나와야 한다."""
    y, p = [], []
    others = [c for c in range(8) if c != NEU]
    for i in range(700):
        second = others[i % len(others)]  # 일곱 후보에 정확히 고르게
        y.append(NEU)
        p.append(dist({NEU: 0.9, second: 0.07, _third(NEU, second): 0.03}))
    rows = A.runner_up_rows(np.array(y), np.stack(p), NAMES)

    assert rows[0]["above_chance"] is False
    assert rows[0]["share"] < 0.25


def test_concentrated_runner_up_alone_is_not_evidence():
    """이 테스트가 P4 의 존재 이유다.

    처음 P4 는 "2순위가 이웃 클래스인가"를 우연(1/7)과 비교했고 **여덟 클래스가
    전부 통과했다.** platelet 은 2순위가 96% 확률로 erythroblast 인데 한 장도
    안 틀린다. 2순위에 실린 확률이 0 에 가깝기 때문이다.

    아래 데이터는 그 상황을 그대로 만든 것이다 — 삼각지대 밖 클래스도 2순위가
    100% 한 클래스로 쏠려 있지만(우연 초과), 실린 확률은 0.001 이다. 옛 기준은
    통과시키고 새 기준은 걸러내야 한다.
    """
    y, p = [], []
    for _ in range(300):  # 삼각지대: 2순위가 실제로 경쟁한다
        y.append(NEU)
        p.append(dist({NEU: 0.80, IG: 0.15, 0: 0.05}))
    for _ in range(300):  # 삼각지대 밖: 2순위가 쏠려 있지만 잔여물이다
        y.append(7)
        p.append(dist({7: 0.998, 2: 0.001, 1: 0.001}))
    y, probs = np.array(y), np.stack(p)

    rows = {r["class"]: r for r in A.runner_up_rows(y, probs, NAMES)}
    # 옛 기준으로는 둘 다 "우연 초과" — 구분이 안 된다
    assert rows[NAMES[NEU]]["above_chance"] is True
    assert rows[NAMES[7]]["above_chance"] is True
    # 새 기준은 실린 확률로 구분한다
    assert rows[NAMES[NEU]]["runner_up_prob"] > 0.1
    assert rows[NAMES[7]]["runner_up_prob"] < 0.01

    check = A.analyze(y, probs, NAMES).verdict["checks"]["P4_2순위에_실린_확률"]
    assert check["passed"] is True


def test_wilson_interval_stays_inside_zero_and_one():
    """비율이 1 일 때 정규근사는 구간이 1 을 넘어간다. 윌슨은 안 넘어간다."""
    lo, hi = A._proportion_ci(50, 50)
    assert 0.0 <= lo <= 1.0 and hi == 1.0
    lo, hi = A._proportion_ci(0, 50)
    assert lo == 0.0 and 0.0 <= hi <= 1.0


def test_pair_table_flags_triangle_pairs():
    y, probs = _dataset(tri_conf=0.55, other_conf=0.95)
    r = A.analyze(y, probs, NAMES)

    flagged = [p for p in r.pairs if p["both_in_triangle"]]
    assert len(flagged) == 1

    tri = flagged[0]
    assert (tri["true"], tri["pred"]) == (NAMES[NEU], NAMES[IG])
    assert tri["count"] == 10
    assert 0.5 < tri["mean_confidence"] < 0.6

    outside = next(p for p in r.pairs if not p["both_in_triangle"])
    assert outside["mean_confidence"] > tri["mean_confidence"]


def test_analysis_is_json_serialisable():
    import json

    y, probs = _dataset(tri_conf=0.55, other_conf=0.95)
    json.dumps(A.analyze(y, probs, NAMES).to_dict(), ensure_ascii=False)
