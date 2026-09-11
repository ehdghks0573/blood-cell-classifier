"""라벨이 반대인 두 무리를 나란히 놓는 그림의 전제 조건.

이 그림이 말하려는 것은 **"양방향이 같은 자리에 있다"** 이고, 그 힘은 전적으로
두 방향이 **둘 다 있다**는 데서 나온다. 한쪽이 비어 있는데도 그림이 나오면,
보는 사람은 한 방향만 보고 "양방향이 같다"고 읽게 된다. 그래서 여기서 못박는
것은 그림의 모양이 아니라 **언제 그리기를 거부하는가**다.
"""

import json

import numpy as np
import pytest

from src.cli import UserError
from src.data import CLASS_NAMES
from tools.boundary import main, resolve

NEU = "neutrophil"
IG = "immature granulocytes"


def case(index, true, pred, conf=0.9):
    return {"index": index, "true": true, "pred": pred, "min_confidence": conf}


def make_data(tmp_path, cases, n=40):
    """npz 한 벌과 consensus JSON 한 벌. 그림이 돌 만큼만 만든다."""
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8)
    # 클래스를 골고루 깔아 둬야 대표를 뽑을 수 있다.
    lbls = np.array([i % len(CLASS_NAMES) for i in range(n)], dtype=np.int64)
    np.savez(tmp_path / "bloodmnist.npz",
             **{f"{s}_{k}": v for s in ("train", "val", "test")
                for k, v in (("images", imgs), ("labels", lbls))})

    path = tmp_path / "consensus.json"
    path.write_text(json.dumps({"runs": ["a", "b"], "split": "test", "n": n,
                                "all_wrong": len(cases), "cases": cases}),
                    encoding="utf-8")
    return str(path)


def argv(tmp_path, consensus, run="r"):
    return ["--run", run, "--data-root", str(tmp_path), "--size", "28",
            "--consensus", consensus, "--pair", "neutrophil", "immature_granulocytes",
            "--out", str(tmp_path / "boundary.png"), "--cpu"]


# ── 클래스 이름 받아들이기 ──

def test_밑줄을_공백으로_읽는다():
    """명령줄에 공백이 든 클래스 이름을 넣기 번거로우므로 밑줄을 허용한다."""
    assert resolve("immature_granulocytes") == IG
    assert resolve("NEUTROPHIL") == NEU


def test_없는_클래스는_있는_것을_알려_준다():
    with pytest.raises(UserError, match="클래스가 아닙니다"):
        resolve("lymphoblast")


# ── 그리기를 거부해야 하는 때 ──

def test_한_방향만_있으면_거부한다(tmp_path):
    """이 그림의 주장은 양방향 대칭이다. 한쪽만으로 그리면 주장이 거짓이 된다."""
    consensus = make_data(tmp_path, [case(1, NEU, IG), case(2, NEU, IG)])

    with pytest.raises(UserError, match="두 방향이 다 있어야"):
        main(argv(tmp_path, consensus))


def test_같은_클래스_둘을_주면_거부한다(tmp_path):
    consensus = make_data(tmp_path, [case(1, NEU, IG), case(2, IG, NEU)])
    args = argv(tmp_path, consensus)
    args[args.index("--pair") + 2] = "neutrophil"

    with pytest.raises(UserError, match="서로 다른 두 클래스"):
        main(args)


def test_없는_consensus_파일은_만드는_법을_알려_준다(tmp_path):
    make_data(tmp_path, [case(1, NEU, IG), case(2, IG, NEU)])

    with pytest.raises(UserError, match="consensus_errors.py"):
        main(argv(tmp_path, str(tmp_path / "없음.json")))


# ── 양방향이 다 있을 때 ──

def test_양방향이_있으면_그린다(tmp_path, monkeypatch):
    """모델 파일 없이도 그림 부분이 도는지 본다 — 실제 학습 결과는 확신 값만 쓴다."""
    import tools.boundary as mod
    monkeypatch.setattr(mod, "load_run", lambda *a, **k: (None, {"input_size": 112}))

    consensus = make_data(tmp_path, [case(1, NEU, IG, 0.95), case(9, NEU, IG, 0.6),
                                     case(2, IG, NEU, 0.8), case(10, IG, NEU, 0.55)])
    assert main(argv(tmp_path, consensus)) == 0
    assert (tmp_path / "boundary.png").stat().st_size > 0
