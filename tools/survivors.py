"""공통 오분류가 **다음 조건에서도 살아남는가**.

    python tools/survivors.py --before runs/consensus_errors.json \
                              --after  runs/consensus_errors_hires.json

`tools/consensus_errors.py` 는 "여러 모델이 같은 장을 틀린다"까지 말한다.
그 결과에는 아직 반론이 하나 남아 있었다.

> 다섯 모델이 시드도 구조도 입력 크기도 다르지만, **원본 픽셀은 다섯 다
> 같은 28px 파일**이다. 28px 로 뭉갠 탓에 사라진 단서가 있다면, 26장은
> 라벨의 문제가 아니라 **해상도의 문제**다.

원본 224px 로 다시 학습한 모델을 넣어 그 반론을 닫는다. 살아남으면 원인은
데이터의 해상도가 아니라 그 이미지 자체에 있다. 대부분 사라지면 26장은
저해상도의 산물이었고, `docs/results.md` 5절의 결론을 고쳐야 한다.

**두 JSON 이 같은 분할·같은 길이여야 한다.** 인덱스로 짝짓기 때문이다.
해상도가 섞였다면 `tools/check_alignment.py` 를 먼저 통과해야 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import UserError, guard  # noqa: E402
from src.console import make_output_encodable  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="공통 오분류의 생존 여부")
    p.add_argument("--before", required=True, help="모델을 넣기 전 consensus JSON")
    p.add_argument("--after", required=True, help="모델을 넣은 뒤 consensus JSON")
    p.add_argument("--save", help="결과 JSON 경로")
    return p.parse_args(argv)


def load(path: str) -> dict:
    f = Path(path)
    if not f.exists():
        raise UserError(f"{f} 가 없습니다. tools/consensus_errors.py --save 로 먼저 만드세요.")
    data = json.loads(f.read_text(encoding="utf-8"))
    if "all_wrong_indices" not in data:
        raise UserError(
            f"{f} 에 all_wrong_indices 가 없습니다. 인덱스를 남기기 전에 만든 파일입니다. "
            "tools/consensus_errors.py 를 --save 로 다시 돌리세요.")
    return data


def label_of(data: dict, index: int) -> tuple[str, str] | None:
    """그 인덱스의 (정답, 예측). cases 에 없으면 방향이 갈린 장이다."""
    for c in data.get("cases", []):
        if c["index"] == index:
            return c["true"], c["pred"]
    return None


def main(argv: list[str] | None = None) -> int:
    make_output_encodable()
    args = parse_args(argv)

    before, after = load(args.before), load(args.after)

    if before["split"] != after["split"] or before["n"] != after["n"]:
        raise UserError(
            f"두 결과의 대상이 다릅니다 "
            f"({before['split']} {before['n']}장 vs {after['split']} {after['n']}장). "
            "같은 분할끼리만 비교할 수 있습니다.")

    added = [r for r in after["runs"] if r not in before["runs"]]
    if not added:
        raise UserError("--after 에 새로 들어간 모델이 없습니다. 비교할 것이 없습니다.")

    old = set(before["all_wrong_indices"])
    new = set(after["all_wrong_indices"])
    survived = sorted(old & new)
    fixed = sorted(old - new)

    print(f"기준  {len(before['runs'])}개 모델이 모두 틀린 장 : {len(old)}장")
    print(f"추가  {', '.join(added)}")
    print(f"결과  {len(after['runs'])}개 모델이 모두 틀린 장 : {len(new)}장")
    print()
    rate = len(survived) / len(old) if old else 0.0
    print(f"살아남음 {len(survived)}장 ({rate:.0%})  ·  새 모델이 맞힌 장 {len(fixed)}장")
    print()

    if survived:
        print("살아남은 장 — 원본 해상도를 올려도 전부 틀린다")
        for i in survived:
            pair = label_of(after, i) or label_of(before, i)
            desc = f"{pair[0]} → {pair[1]}" if pair else "(모델마다 다른 방향으로 틀림)"
            print(f"  #{i:<5d} {desc}")
        print()

    if fixed:
        print("새 모델이 맞힌 장 — 저해상도 탓이었던 것")
        for i in fixed:
            pair = label_of(before, i)
            desc = f"{pair[0]} → {pair[1]}" if pair else "(모델마다 다른 방향으로 틀림)"
            print(f"  #{i:<5d} {desc}")
        print()

    # 해석은 사람이 하지만, 어느 쪽 이야기인지는 숫자가 정한다.
    if rate >= 0.8:
        print("→ 거의 그대로 남는다. 원인은 해상도가 아니라 그 이미지에 있다.")
    elif rate <= 0.3:
        print("→ 대부분 사라진다. 저해상도의 산물이었다는 뜻이므로 결론을 고쳐야 한다.")
    else:
        print("→ 절반쯤 남는다. 해상도로 설명되는 몫과 남는 몫이 섞여 있다.")

    if args.save:
        out = {
            "before": {"runs": before["runs"], "all_wrong": len(old)},
            "after": {"runs": after["runs"], "all_wrong": len(new)},
            "added": added,
            "split": after["split"],
            "n": after["n"],
            "survived": survived,
            "fixed": fixed,
            "survival_rate": round(rate, 4),
        }
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
