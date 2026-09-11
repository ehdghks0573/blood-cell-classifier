"""공통 오분류의 생존 판정.

`tools/survivors.py` 는 JSON 두 개를 인덱스로 짝짓는다. 여기서 조용히 틀리면
"26장 중 24장이 살아남았다" 같은 문장이 아무 근거 없이 나온다. 숫자가 결론을
그대로 정하는 자리라서, 계산보다 **엉뚱한 짝짓기를 거부하는가**를 더 본다.
"""

import json

import pytest

from src.cli import UserError
from tools.survivors import main


def consensus(tmp_path, name, runs, indices, cases=None, split="test", n=3421,
              with_indices=True):
    """consensus_errors.py 가 --save 로 남기는 모양의 JSON 을 만든다."""
    data = {
        "runs": runs,
        "split": split,
        "n": n,
        "all_wrong": len(indices),
        "cases": cases or [],
    }
    if with_indices:
        data["all_wrong_indices"] = list(indices)
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_살아남은_장과_맞힌_장을_가른다(tmp_path, capsys):
    before = consensus(tmp_path, "before.json", ["a", "b"], [3, 7, 11, 20])
    after = consensus(tmp_path, "after.json", ["a", "b", "hi"], [7, 20])
    out = tmp_path / "survivors.json"

    assert main(["--before", before, "--after", after, "--save", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["survived"] == [7, 20]
    assert saved["fixed"] == [3, 11]
    assert saved["added"] == ["hi"]
    assert saved["survival_rate"] == 0.5
    assert "절반쯤 남는다" in capsys.readouterr().out


def test_전부_살아남으면_이미지_탓이라고_읽는다(tmp_path, capsys):
    before = consensus(tmp_path, "before.json", ["a"], [1, 2, 3, 4, 5])
    after = consensus(tmp_path, "after.json", ["a", "hi"], [1, 2, 3, 4, 5])

    assert main(["--before", before, "--after", after]) == 0
    assert "해상도가 아니라 그 이미지에 있다" in capsys.readouterr().out


def test_대부분_사라지면_결론을_고치라고_말한다(tmp_path, capsys):
    before = consensus(tmp_path, "before.json", ["a"], [1, 2, 3, 4, 5])
    after = consensus(tmp_path, "after.json", ["a", "hi"], [3])

    assert main(["--before", before, "--after", after]) == 0
    assert "결론을 고쳐야 한다" in capsys.readouterr().out


def test_정답_예측_클래스를_붙여_보여_준다(tmp_path, capsys):
    cases = [{"index": 7, "true": "neutrophil", "pred": "immature granulocytes",
              "min_confidence": 0.99}]
    before = consensus(tmp_path, "before.json", ["a"], [7, 9], cases=cases)
    after = consensus(tmp_path, "after.json", ["a", "hi"], [7], cases=cases)

    main(["--before", before, "--after", after])
    out = capsys.readouterr().out
    assert "neutrophil → immature granulocytes" in out
    # 9번은 cases 에 없다 — 모델마다 다른 방향으로 틀린 장이라 이름을 못 붙인다.
    assert "모델마다 다른 방향으로 틀림" in out


def test_다른_분할끼리는_비교를_거부한다(tmp_path):
    before = consensus(tmp_path, "before.json", ["a"], [1], split="val", n=1712)
    after = consensus(tmp_path, "after.json", ["a", "hi"], [1], split="test", n=3421)

    with pytest.raises(UserError, match="같은 분할"):
        main(["--before", before, "--after", after])


def test_장수가_다르면_거부한다(tmp_path):
    before = consensus(tmp_path, "before.json", ["a"], [1], n=3421)
    after = consensus(tmp_path, "after.json", ["a", "hi"], [1], n=3000)

    with pytest.raises(UserError):
        main(["--before", before, "--after", after])


def test_새_모델이_없으면_거부한다(tmp_path):
    """같은 모델 목록끼리 비교하면 '살아남았다'가 아무 뜻도 없다."""
    before = consensus(tmp_path, "before.json", ["a", "b"], [1, 2])
    after = consensus(tmp_path, "after.json", ["b", "a"], [1])

    with pytest.raises(UserError, match="새로 들어간 모델이 없습니다"):
        main(["--before", before, "--after", after])


def test_인덱스가_없는_옛_파일은_안내한다(tmp_path):
    """all_wrong_indices 는 나중에 추가됐다. 그 전 파일을 조용히 0장으로 읽으면 안 된다."""
    before = consensus(tmp_path, "before.json", ["a"], [1, 2], with_indices=False)
    after = consensus(tmp_path, "after.json", ["a", "hi"], [1])

    with pytest.raises(UserError, match="all_wrong_indices"):
        main(["--before", before, "--after", after])


def test_없는_파일은_만드는_법을_알려_준다(tmp_path):
    after = consensus(tmp_path, "after.json", ["a", "hi"], [1])

    with pytest.raises(UserError, match="consensus_errors.py"):
        main(["--before", str(tmp_path / "없음.json"), "--after", after])
