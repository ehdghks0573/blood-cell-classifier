"""발표 슬라이드를 한 파일로 묶는다 (Day 14).

    python tools/build_slides.py

`docs/slides.src.html` 의 자리표시자를 실제 그림으로 바꿔 `docs/slides.html`
을 만든다. 그림은 data URI 로 파일 안에 들어가므로, 결과물 하나만 있으면
어디서든 열린다 — 발표장에서 경로가 깨질 일이 없다.

**그림을 손으로 복사해 넣지 않는 이유.** 실험을 다시 돌리면 혼동행렬도
히트맵도 바뀐다. 그때 슬라이드가 옛 그림을 들고 있으면 발표에서 틀린 숫자를
말하게 된다. 이 스크립트를 다시 돌리면 최신 결과로 갱신된다.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

#: 자리표시자 → (파일, 가로 최대 픽셀, 형식)
#: 사진 성격인 것은 JPEG 로 줄이고, 글자가 든 그림은 PNG 로 둔다.
#: JPEG 로 글자를 누르면 축 라벨과 숫자가 뭉개진다.
FIGURES = {
    "IMG_CONFUSION": ("runs/{run}/confusion_test.png",                 1000, "PNG"),
    "IMG_GRADCAM":   ("runs/{run}/gradcam_classes.png",                1150, "JPEG"),
    "IMG_ERRORS":    ("runs/{run}/errors/"
                      "neutrophil__to__immature_granulocytes.png",     1500, "JPEG"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="발표 슬라이드 빌드")
    p.add_argument("--run", default="effb0_112",
                   help="그림을 가져올 실행 이름 (기본: 발표에 쓰는 최고 성능 모델)")
    p.add_argument("--src", default=str(ROOT / "docs" / "slides.src.html"))
    p.add_argument("--out", default=str(ROOT / "docs" / "slides.html"))
    p.add_argument("--figures", default=str(ROOT / "docs" / "figures"),
                   help="세포 낱장이 있는 폴더. 점검용으로 돌릴 때 따로 둔다")
    p.add_argument("--quality", type=int, default=88, help="JPEG 품질")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    """보여 주기용 경로. 프로젝트 밖이거나 상대 경로면 그대로 둔다.

    `relative_to` 는 밖에 있는 경로에 대해 예외를 던진다. 출력 경로를
    상대 경로로 주면 마지막 줄에서 터졌다 — 결과물은 다 만들어 놓고
    성공 메시지를 찍다가 실패하는, 가장 헷갈리는 종류의 실패였다.
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def data_uri(path: Path, max_width: int, fmt: str, quality: int) -> tuple[str, int]:
    """이미지를 줄여 data URI 로 만든다. (uri, 킬로바이트) 를 돌려준다."""
    img = Image.open(path)
    if img.width > max_width:
        height = round(img.height * max_width / img.width)
        img = img.resize((max_width, height), Image.LANCZOS)

    buf = io.BytesIO()
    if fmt == "JPEG":
        # 투명 배경이 있으면 흰색 위에 얹는다. 안 그러면 검게 나온다.
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            flat = Image.new("RGB", img.size, (255, 255, 255))
            flat.paste(img, mask=img.split()[-1])
            img = flat
        img.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"

    raw = buf.getvalue()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw) // 1024


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.src)
    if not src.exists():
        print(f"{src} 가 없습니다.")
        return 1

    html = src.read_text(encoding="utf-8")
    missing = []

    # 세포 낱장 — docs/figures/<이름>.png 가 {{CELL_<이름>}} 이 된다.
    # 목록을 손으로 관리하면 tools/extract_cells.py 가 새 이미지를 뽑을 때마다
    # 여기도 고쳐야 한다. 폴더를 그대로 읽는다.
    cells = sorted(Path(args.figures).glob("*.png"))
    used = 0
    for path in cells:
        placeholder = "{{CELL_" + path.stem + "}}"
        if placeholder not in html:
            continue
        # 28px 원본을 정수배로 키운 그림이라 줄이지 않는다. 부드럽게 늘리거나
        # JPEG 로 누르면 핵이 나뉘었는지가 흐려진다 — 그게 발표의 핵심이다.
        uri, kb = data_uri(path, 10_000, "PNG", args.quality)
        html = html.replace(placeholder, uri)
        used += 1
    if used:
        print(f"  세포 낱장 {used}장 ({rel(Path(args.figures))})")

    for token, (template, width, fmt) in FIGURES.items():
        path = ROOT / template.format(run=args.run)
        placeholder = "{{" + token + "}}"
        if placeholder not in html:
            print(f"경고: {placeholder} 가 원본에 없습니다")
            continue
        if not path.exists():
            missing.append(rel(path))
            continue

        uri, kb = data_uri(path, width, fmt, args.quality)
        html = html.replace(placeholder, uri)
        print(f"  {token:<14s} {rel(path)}  →  {kb}KB ({fmt})")

    if missing:
        print("\n그림이 없습니다:")
        for m in missing:
            print(f"  {m}")
        print("\nevaluate.py --save, explain.py, inspect_errors.py 를 먼저 실행하세요.")
        return 1

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) // 1024
    print(f"\n{rel(out)}  {size_kb:,}KB")
    if size_kb > 15000:
        print("경고: 16MB 에 가깝습니다. --quality 를 낮추거나 가로 폭을 줄이세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
