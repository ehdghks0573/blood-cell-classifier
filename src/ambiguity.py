"""라벨 모호성 가설의 검증 지표.

`docs/results.md` 4절에서 이런 가설을 세웠다 — 남은 오분류가 세 클래스
(neutrophil, immature granulocytes, monocyte) 안에만 갇혀 있는 이유는 모델의
결함이 아니라 **정답 라벨 자체가 연속선상에 있어서 모호하기 때문**이다.

가설은 세워 두면 그냥 이야기일 뿐이고, **틀릴 수 있는 형태로 적어야 검증이다.**
라벨이 모호하다면 모델의 확률 출력이 다음과 같아야 한다.

1. 삼각지대 오류에서 모델이 **덜 확신한다** — 확률이 정답과 예측에 나뉜다
2. 삼각지대 오류에서 **정답이 2순위**로 남아 있는 비율이 높다
3. 삼각지대 오류의 **마진**(p_예측 − p_정답)이 그 밖의 오류보다 작다

셋 다 반대로 나오면(확신에 차서 틀리고, 정답 확률이 0에 가깝다면) 가설은 틀린
것이고, 그건 모호성이 아니라 모델이 특징을 못 잡은 것이다. **어느 쪽이든 결과다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

#: 혼동이 갇혀 있는 세 클래스. `docs/results.md` 4절 관찰 1.
#: 미성숙 과립구는 성숙하면 호중구가 되고, 단핵구는 그 둘과 형태가 겹친다.
TRIANGLE = (3, 5, 6)  # immature granulocytes, monocyte, neutrophil

#: 이 이상이면 "확신에 차서 틀렸다"고 본다. 모호성 가설의 반증 사례다.
CONFIDENT_WRONG = 0.90

#: 묶음이 이보다 작으면 비교 자체를 하지 않는다.
MIN_GROUP = 5


def entropy(probs: np.ndarray) -> np.ndarray:
    """행마다 예측 분포의 엔트로피. 높을수록 모델이 갈피를 못 잡은 것이다.

    0 을 작은 값으로 잘라내는 대신 항 자체를 0 으로 둔다. 0·log0 은 극한에서
    0 이고, 잘라내면 확신에 찬 예측의 엔트로피가 0 이 아니게 나온다.
    """
    p = np.asarray(probs, dtype=np.float64)
    terms = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=1)


@dataclass
class GroupStats:
    """한 묶음(정답 / 삼각지대 오류 / 그 밖의 오류)의 확신도 요약."""

    name: str
    n: int
    mean_confidence: float  # 예측 클래스에 준 확률의 평균
    median_confidence: float
    mean_entropy: float
    mean_true_prob: float  # 정답 클래스에 준 확률의 평균
    mean_margin: float  # p_예측 − p_정답. 오류에서만 의미가 있다
    true_in_top2: float  # 정답이 1·2순위 안에 있는 비율
    confident_wrong: int  # 확률 0.90 이상으로 틀린 장수

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "mean_confidence": round(self.mean_confidence, 4),
            "median_confidence": round(self.median_confidence, 4),
            "mean_entropy": round(self.mean_entropy, 4),
            "mean_true_prob": round(self.mean_true_prob, 4),
            "mean_margin": round(self.mean_margin, 4),
            "true_in_top2": round(self.true_in_top2, 4),
            "confident_wrong": self.confident_wrong,
        }


def sample_vectors(y_true: np.ndarray, probs: np.ndarray,
                   mask: np.ndarray) -> dict[str, np.ndarray]:
    """`mask` 로 고른 표본들의 **장당** 값. 평균을 내기 전 단계다.

    평균만 갖고 있으면 표본 오차를 계산할 수 없다. 부트스트랩으로 불확실성을
    재려면 원래 값이 필요하므로 여기서 한 번만 뽑아 두고 돌려 쓴다.
    """
    idx = np.flatnonzero(mask)
    p = probs[idx]
    t = y_true[idx]
    if idx.size == 0:
        empty = np.zeros(0)
        return {"confidence": empty, "true_prob": empty, "margin": empty,
                "in_top2": empty, "entropy": empty, "runner_up_prob": empty,
                "wrong": np.zeros(0, dtype=bool)}

    pred = p.argmax(axis=1)
    conf = p[np.arange(len(idx)), pred]
    true_p = p[np.arange(len(idx)), t]
    order = np.argsort(-p, axis=1)
    top2 = order[:, :2]

    return {
        "confidence": conf,
        "true_prob": true_p,
        "margin": conf - true_p,
        "in_top2": (top2 == t[:, None]).any(axis=1).astype(float),
        "entropy": entropy(p),
        "wrong": (pred != t),
        # 2순위에 실제로 얼마나 확률을 줬는가. **어느 클래스가 2순위인가와는
        # 전혀 다른 이야기다.** platelet 의 2순위는 96% 가 erythroblast 지만
        # 거기 실린 확률은 0 에 가깝다 — 경쟁이 아니라 잔여물이다.
        "runner_up_prob": p[np.arange(len(idx)), order[:, 1]],
    }


def group_stats(name: str, y_true: np.ndarray, probs: np.ndarray,
                mask: np.ndarray) -> GroupStats:
    """`mask` 로 고른 표본들의 확신도 통계."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return GroupStats(name, 0, *(float("nan"),) * 6, 0)

    v = sample_vectors(y_true, probs, mask)

    return GroupStats(
        name=name,
        n=int(idx.size),
        mean_confidence=float(v["confidence"].mean()),
        median_confidence=float(np.median(v["confidence"])),
        mean_entropy=float(v["entropy"].mean()),
        mean_true_prob=float(v["true_prob"].mean()),
        mean_margin=float(v["margin"].mean()),
        true_in_top2=float(v["in_top2"].mean()),
        confident_wrong=int((v["wrong"] & (v["confidence"] >= CONFIDENT_WRONG)).sum()),
    )


def in_triangle(labels: np.ndarray, triangle: tuple[int, ...] = TRIANGLE) -> np.ndarray:
    return np.isin(labels, np.asarray(triangle))


@dataclass
class Analysis:
    """가설 검증 결과 전체."""

    groups: list[GroupStats]
    pairs: list[dict] = field(default_factory=list)
    runner_ups: list[dict] = field(default_factory=list)
    verdict: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "pairs": self.pairs,
            "runner_ups": self.runner_ups,
            "verdict": self.verdict,
        }


def analyze(y_true: np.ndarray, probs: np.ndarray, class_names: list[str],
            triangle: tuple[int, ...] = TRIANGLE) -> Analysis:
    """세 묶음으로 나눠 비교하고, 가설의 세 예측을 하나씩 판정한다.

    묶음을 나누는 기준은 **정답과 예측이 모두 삼각지대 안인가**이다. 정답만
    삼각지대이고 예측이 밖으로 나갔다면 그건 연속선상의 혼동이 아니라 다른
    종류의 오류이므로 '그 밖'으로 센다.
    """
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)

    wrong = y_pred != y_true
    tri = in_triangle(y_true, triangle) & in_triangle(y_pred, triangle)

    groups = [
        group_stats("정답", y_true, probs, ~wrong),
        group_stats("삼각지대 오류", y_true, probs, wrong & tri),
        group_stats("그 밖의 오류", y_true, probs, wrong & ~tri),
    ]

    pairs = _pair_table(y_true, y_pred, probs, class_names, wrong)
    correct, tri_err, other_err = groups

    # 표본이 적으면 비교 자체가 성립하지 않는다. 판정을 미루는 것도 결과다.
    enough = tri_err.n >= MIN_GROUP and other_err.n >= MIN_GROUP

    tri_v = sample_vectors(y_true, probs, wrong & tri)
    other_v = sample_vectors(y_true, probs, wrong & ~tri)

    checks = {
        "P1_덜_확신한다": _check(
            enough, tri_v["confidence"], other_v["confidence"], expect="less", fmt="{:.3f}"),
        "P2_정답이_2순위": _check(
            enough, tri_v["in_top2"], other_v["in_top2"], expect="greater", fmt="{:.1%}"),
        "P3_마진이_작다": _check(
            enough, tri_v["margin"], other_v["margin"], expect="less", fmt="{:.3f}"),
    }

    runner_ups = runner_up_rows(y_true, probs, class_names)

    # P4 는 오분류가 아니라 정답 수천 장으로 재는 항목이라, 앞의 셋과 달리
    # 표본 부족으로 판정을 미룰 일이 거의 없다.
    # P4 는 맞힌 장 수천 개로 재므로 표본이 부족할 일이 없다.
    #
    # **기준을 한 번 갈아엎었다.** 처음에는 "2순위가 이웃 클래스인가"를 우연
    # (1/7) 과 비교했는데, 여덟 클래스가 전부 통과했다. platelet 은 2순위가
    # 96% 확률로 erythroblast 인데 실제로는 한 장도 안 틀린다. 어느 클래스가
    # 2순위인지는 모든 클래스에서 쏠려 나오므로 **구분에 쓸 수 없는 숫자**다.
    # 항상 통과하는 검사는 검사가 아니다.
    #
    # 실제로 다른 것은 2순위에 실린 **확률의 크기**다. 라벨이 연속선상에
    # 있다면 이웃 클래스가 실질적으로 경쟁해야 하고, 그렇지 않다면 2순위는
    # 0 에 가까운 잔여물이다.
    tri_correct = sample_vectors(y_true, probs, ~wrong & in_triangle(y_true, triangle))
    other_correct = sample_vectors(y_true, probs, ~wrong & ~in_triangle(y_true, triangle))
    checks["P4_2순위에_실린_확률"] = _check(
        tri_correct["runner_up_prob"].size >= MIN_GROUP
        and other_correct["runner_up_prob"].size >= MIN_GROUP,
        tri_correct["runner_up_prob"], other_correct["runner_up_prob"],
        expect="greater", fmt="{:.4f}")

    passed = sum(1 for c in checks.values() if c["passed"] is True)
    failed = sum(1 for c in checks.values() if c["passed"] is False)
    verdict = {
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "undecided": len(checks) - passed - failed,
        "total": len(checks),
        "comparable": enough,
        "summary": _summary(passed, failed, len(checks), enough, correct, tri_err),
    }
    return Analysis(groups=groups, pairs=pairs, runner_ups=runner_ups, verdict=verdict)


def _check(comparable: bool, tri: np.ndarray, other: np.ndarray,
           expect: str, fmt: str) -> dict:
    """두 묶음의 평균 차이를 **불확실성과 함께** 판정한다.

    부등호 하나로 비교하면 안 된다. 여기서 다루는 묶음은 30~50장짜리라
    평균이 표본에 따라 쉽게 0.05 씩 움직인다. 실제로 처음에 부등호로 짰다가,
    같은 데이터로 학습한 두 모델이 정반대 판정(1/3 과 3/3)을 내는 것을 보고
    이 방식으로 바꿨다.

    부트스트랩으로 차이의 95% 구간을 구해, 구간이 0 을 걸치면 **판정을 보류**한다.
    "차이가 없다"가 아니라 "이 표본으로는 말할 수 없다"는 뜻이다.
    """
    if not comparable or tri.size == 0 or other.size == 0:
        return {"passed": None, "evidence": "표본 부족"}

    diff = float(tri.mean() - other.mean())
    lo, hi = _bootstrap_ci(tri, other)

    if lo > 0:
        decided = "greater"
    elif hi < 0:
        decided = "less"
    else:
        decided = None

    passed = None if decided is None else (decided == expect)
    evidence = (f"삼각지대 {fmt.format(tri.mean())} vs 그 밖 {fmt.format(other.mean())} · "
                f"차이 {diff:+.3f} (95% 구간 {lo:+.3f} ~ {hi:+.3f})")
    return {"passed": passed, "diff": round(diff, 4),
            "ci": [round(lo, 4), round(hi, 4)], "evidence": evidence}


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                  seed: int = 0) -> tuple[float, float]:
    """평균 차이(a − b)의 95% 부트스트랩 구간.

    시드를 고정한다. 판정이 실행할 때마다 바뀌면 그것도 재현성 문제다.
    """
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _summary(passed: int, failed: int, total: int, comparable: bool,
             correct: GroupStats, tri: GroupStats) -> str:
    if not comparable:
        return "표본이 적어 판정을 미룬다. 오분류가 그만큼 적다는 뜻이기도 하다."

    undecided = total - passed - failed
    decided = passed + failed
    base = (f"삼각지대 오류에서 확신도는 {correct.mean_confidence:.3f} → "
            f"{tri.mean_confidence:.3f} 로 떨어지고, 정답이 2순위에 남는 비율은 "
            f"{tri.true_in_top2:.0%} 다.")
    tail = f" (판정보류 {undecided}개는 표본이 모자라 가릴 수 없었다)" if undecided else ""

    if decided == 0:
        return (f"{base} {total}개 항목이 모두 판정보류다. **판정할 수 없다.** "
                "오분류가 수십 장뿐이라 이 표본으로는 차이를 가릴 수 없다는 뜻이다.")
    if failed == 0:
        return (f"{base} 판정된 {decided}개가 모두 맞았다 — 라벨 모호성 가설과 "
                f"일치한다.{tail}")
    if passed == 0:
        return (f"{base} 그러나 판정된 {decided}개가 모두 반대로 나왔다. 모호성이 "
                f"아니라 모델이 특징을 못 잡은 것으로 봐야 한다.{tail}")
    return (f"{base} 맞음 {passed} · 빗나감 {failed} · 판정보류 {undecided}. "
            "가설을 그대로 받아들일 수 없다. 어긋난 항목을 그대로 적고 다시 본다.")


def runner_up_rows(y_true: np.ndarray, probs: np.ndarray,
                   class_names: list[str]) -> list[dict]:
    """**맞힌** 예측에서 2순위가 어느 클래스였는지.

    오분류만 보면 표본이 수십 장이라 아무것도 가릴 수 없다. 반면 정답은 수천
    장이다. 라벨이 연속선상에 있다면 그 이웃 관계는 틀린 장에서만 나타날 리가
    없고, **맞힌 장에서도 2순위로 드러나야 한다.** 같은 가설을 100배 큰 표본으로
    확인하는 셈이다.

    비교 기준은 우연히 그 클래스가 2순위가 될 확률, 즉 1/(클래스 수 − 1) 이다.
    """
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)
    k = len(class_names)
    chance = 1.0 / (k - 1)

    rows = []
    for c in range(k):
        mask = (y_true == c) & (y_pred == c)
        n = int(mask.sum())
        if n == 0:
            continue

        ordered = np.argsort(-probs[mask], axis=1)
        second = ordered[:, 1]
        counts = np.bincount(second, minlength=k)
        top = int(counts.argmax())
        share = counts[top] / n
        lo, hi = _proportion_ci(counts[top], n, z=_z_bonferroni(k - 1))
        mass = probs[mask][np.arange(n), second]

        rows.append({
            "class": class_names[c],
            "n_correct": n,
            "runner_up": class_names[top],
            "share": round(float(share), 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "chance": round(chance, 4),
            "above_chance": bool(lo > chance),
            # 이 열이 실제 신호다. 2순위가 어느 클래스인지는 모든 클래스에서
            # 쏠려 나오므로 구분에 쓸 수 없다.
            "runner_up_prob": round(float(mass.mean()), 5),
            "both_in_triangle": bool(c in TRIANGLE and top in TRIANGLE),
        })
    return rows


def _z_bonferroni(n_categories: int, alpha: float = 0.05) -> float:
    """다중비교 보정된 z 값.

    후보 7개 중 **가장 많이 나온 것**을 골라 놓고 "우연(1/7)보다 큰가"를 물으면
    안 된다. 최댓값은 고르게 흩어져 있어도 1/7 보다 크게 나온다. 후보 수만큼
    유의수준을 나눠(본페로니) 그 편향을 상쇄한다. 7개면 z 는 1.96 이 아니라
    약 2.69 가 된다.
    """
    return NormalDist().inv_cdf(1 - alpha / (2 * n_categories))


def _proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 윌슨 신뢰구간.

    정규근사(p ± z·se)는 비율이 0 이나 1 에 가까울 때 구간이 범위를 벗어난다.
    윌슨 구간은 그런 경우에도 [0, 1] 안에 머문다.
    """
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _pair_table(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                class_names: list[str], wrong: np.ndarray, k: int = 6) -> list[dict]:
    """혼동 쌍마다 몇 장 틀렸고 그때 얼마나 확신했는지."""
    rows: list[dict] = []
    for t in range(len(class_names)):
        for p in range(len(class_names)):
            if t == p:
                continue
            m = wrong & (y_true == t) & (y_pred == p)
            n = int(m.sum())
            if n == 0:
                continue
            idx = np.flatnonzero(m)
            rows.append({
                "true": class_names[t],
                "pred": class_names[p],
                "count": n,
                "mean_confidence": round(float(probs[idx, p].mean()), 4),
                "mean_true_prob": round(float(probs[idx, t].mean()), 4),
                "both_in_triangle": bool(t in TRIANGLE and p in TRIANGLE),
            })
    rows.sort(key=lambda r: -r["count"])
    return rows[:k]
