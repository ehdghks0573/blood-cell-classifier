"""라벨이 **반대인데 생김새가 같은** 무리를 한 판에 놓는다.

    python tools/boundary.py --run hires224_res18_112 --size 224 \
                             --consensus runs/consensus_errors_hires.json \
                             --pair neutrophil immature_granulocytes

`inspect_errors.py` 는 한 쌍을 한 방향으로만 본다 — "호중구인데 미성숙
과립구로 틀린 장"을 두 클래스의 대표와 나란히 놓는다. 그것만으로는 답할 수
없는 질문이 하나 남는다.

> **반대 방향으로 틀린 장들은 어떻게 생겼는가?**

이게 왜 중요한가. 한 방향만 보면 "호중구 라벨이 붙었는데 미성숙해 보인다"
까지 말할 수 있고, 그건 *모델이 미성숙 쪽으로 기울어 있다*로도 설명된다.
그런데 **반대 방향으로 틀린 장들도 똑같이 생겼다면** 그 설명이 무너진다.
모델의 기울기는 한쪽으로만 작용하기 때문이다. 남는 설명은 하나다 —
두 무리가 **같은 경계에 있고, 라벨이 그 경계를 가르지 못한다.**

그래서 두 방향을 위아래 줄로 놓고, 아래에 두 클래스의 대표를 깐다.
보는 사람이 답할 질문은 하나다 — **위 두 줄을 서로 구분할 수 있는가.**
구분이 안 되면, 그 둘을 가른 것은 생김새가 아니라 라벨이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import UserError, guard, load_run, pick_device  # noqa: E402
from src.console import make_output_encodable  # noqa: E402
from src.data import CLASS_NAMES, load_split  # noqa: E402
from src.plots import use_korean_font  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="반대 방향으로 틀린 장들을 나란히 본다")
    p.add_argument("--run", required=True, help="히트맵·확신을 가져올 실행")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=224, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--consensus", required=True, help="consensus_errors.py 가 --save 로 남긴 JSON")
    p.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"),
                   help="클래스 두 개. 밑줄은 공백으로 본다 (immature_granulocytes)")
    p.add_argument("--refs", type=int, default=4, help="대표 이미지 장수")
    p.add_argument("--out", help="저장 경로 (기본: runs/<run>/errors/boundary.png)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def resolve(name: str) -> str:
    """`immature_granulocytes` 처럼 받은 것을 실제 클래스 이름으로 바꾼다."""
    want = name.replace("_", " ").strip().lower()
    for c in CLASS_NAMES:
        if c.lower() == want:
            return c
    raise UserError(f"'{name}' 은 클래스가 아닙니다. 있는 것: {', '.join(CLASS_NAMES)}")


def main(argv: list[str] | None = None) -> int:
    make_output_encodable()
    use_korean_font()
    args = parse_args(argv)

    a, b = resolve(args.pair[0]), resolve(args.pair[1])
    if a == b:
        raise UserError("서로 다른 두 클래스를 주세요. 같은 클래스끼리는 비교가 없습니다.")

    path = Path(args.consensus)
    if not path.exists():
        raise UserError(f"{path} 가 없습니다. tools/consensus_errors.py --save 로 먼저 만드세요.")
    data = json.loads(path.read_text(encoding="utf-8"))

    # 두 방향을 각각 모은다. cases 는 모든 모델이 **같은 답으로** 틀린 장이다.
    forward = [c for c in data.get("cases", []) if c["true"] == a and c["pred"] == b]
    backward = [c for c in data.get("cases", []) if c["true"] == b and c["pred"] == a]
    if not forward or not backward:
        raise UserError(
            f"두 방향이 다 있어야 비교가 됩니다 "
            f"({a}→{b} {len(forward)}장 · {b}→{a} {len(backward)}장). "
            "한쪽이 0장이면 이 그림으로 말할 수 있는 것이 없습니다.")

    for rows in (forward, backward):
        rows.sort(key=lambda r: -r["min_confidence"])

    imgs, lbls = load_split(args.data_root, args.size, args.split)
    lbls = np.asarray(lbls).ravel()

    device = pick_device(args.cpu)
    model, ckpt = load_run(args.run, ROOT, device)
    del model  # 대표를 고르는 데에는 확신만 쓴다. 여기서는 라벨로 충분하다.

    def refs_for(cls: str) -> list[np.ndarray]:
        idx = np.flatnonzero(lbls == CLASS_NAMES.index(cls))
        wrong = {c["index"] for c in data.get("cases", [])}
        clean = [i for i in idx if int(i) not in wrong]
        pick = np.linspace(0, len(clean) - 1, args.refs).astype(int)
        return [imgs[clean[i]] for i in pick]

    ncols = max(len(forward), len(backward), args.refs)
    fig, axes = plt.subplots(4, ncols, figsize=(1.55 * ncols, 6.8))
    if ncols == 1:
        axes = axes.reshape(4, 1)

    band = [
        (f"라벨: {a}  ·  모델의 답: {b}", forward, None),
        (f"라벨: {b}  ·  모델의 답: {a}", backward, None),
        (f"대표 — {a}", None, a),
        (f"대표 — {b}", None, b),
    ]

    for r, (title, rows, cls) in enumerate(band):
        pics = [imgs[c["index"]] for c in rows] if rows is not None else refs_for(cls)
        subs = ([f"#{c['index']}  확신 {c['min_confidence']:.2f}" for c in rows]
                if rows is not None else [""] * len(pics))
        for k in range(ncols):
            ax = axes[r, k]
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if k < len(pics):
                ax.imshow(pics[k])
                if subs[k]:
                    ax.set_xlabel(subs[k], fontsize=7.5)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(title, fontsize=8.5, rotation=0, ha="right", va="center",
                              labelpad=8)

    fig.suptitle(
        f"{a} ↔ {b} — 라벨만 반대인 두 무리\n"
        f"물어볼 것: 위 두 줄을 서로 구분할 수 있는가. 못 하면 그 둘을 가른 것은 생김새가 아니다\n"
        f"(원본 {args.size}px · 모든 모델이 같은 답으로 틀린 장만)",
        fontsize=10)
    fig.tight_layout(rect=(0.06, 0, 1, 0.93))

    out = Path(args.out) if args.out else ROOT / "runs" / args.run / "errors" / "boundary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{a} → {b}  {len(forward)}장")
    print(f"{b} → {a}  {len(backward)}장")
    print(f"입력 {ckpt['input_size']}px · 원본 {args.size}px")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
