"""발표용 세포 낱장 이미지를 뽑는다.

    python tools/extract_cells.py --run effb0_112

격자 그림은 근거를 보여주기에는 좋지만 **처음 보는 사람에게는 빽빽하다.**
슬라이드에서 "이게 호중구, 이게 미성숙 과립구"라고 짚으려면 낱장이 필요하다.

두 종류를 뽑는다.

  class_<이름>.png    클래스마다 모델이 가장 확신 있게 맞힌 한 장 (전형)
  case_<이름>.png     여러 모델이 공통으로 틀린 장 (근거 이미지)

28px 원본을 그대로 키우면 뭉개지므로, **최근접 이웃**으로 정수배 확대한다.
부드럽게 늘리면 없던 경계가 생겨서, 핵이 분엽됐는지 아닌지가 흐려진다.
그 구분이 이 발표의 핵심이라 뭉개면 안 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.console import make_output_encodable  # noqa: E402
from inspect_errors import prototypes  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    BloodDataset,
    build_transform,
    load_split,
)
from src.engine import predict  # noqa: E402
from src.model import build_model  # noqa: E402

SLUG = {
    "basophil": "basophil",
    "eosinophil": "eosinophil",
    "erythroblast": "erythroblast",
    "immature granulocytes": "ig",
    "lymphocyte": "lymphocyte",
    "monocyte": "monocyte",
    "neutrophil": "neutrophil",
    "platelet": "platelet",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="발표용 세포 낱장 추출")
    p.add_argument("--run", default="effb0_112")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--scale", type=int, default=6, help="정수배 확대율")
    p.add_argument("--consensus", default=str(ROOT / "runs" / "consensus_errors.json"),
                   help="공통 오분류 JSON. 없으면 이 모델의 오분류에서 고른다")
    p.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def save(img: np.ndarray, path: Path, scale: int) -> None:
    im = Image.fromarray(img.astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG", optimize=True)


def main(argv: list[str] | None = None) -> int:
    make_output_encodable()
    args = parse_args(argv)
    run_dir = ROOT / "runs" / args.run
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"{ckpt_path} 가 없습니다.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    images, lbls = load_split(args.data_root, args.size, args.split)
    dataset = BloodDataset(images, lbls, build_transform(ckpt["input_size"], train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    y_true, y_pred, probs, _ = predict(model, loader, device)

    out = Path(args.out)
    manifest: dict[str, dict] = {}

    for c, name in enumerate(CLASS_NAMES):
        idx = prototypes(probs, y_true, y_pred, c, n=1)
        if idx.size == 0:
            print(f"  {name}: 맞힌 장이 없어 건너뜁니다")
            continue
        i = int(idx[0])
        path = out / f"class_{SLUG[name]}.png"
        save(images[i], path, args.scale)
        manifest[f"class_{SLUG[name]}"] = {
            "class": name, "index": i,
            "confidence": round(float(probs[i, c]), 4),
        }
        print(f"  {name:<24s} #{i:<5d} 확신 {probs[i, c]:.3f}  {path.name}")

    # 근거 이미지 — 여러 모델이 공통으로 틀린 장을 우선한다
    consensus = Path(args.consensus)
    cases: list[dict] = []
    if consensus.exists():
        data = json.loads(consensus.read_text(encoding="utf-8"))
        cases = data["cases"]
        print(f"\n공통 오분류 {len(cases)}장에서 고릅니다 ({len(data['runs'])}개 모델)")
    else:
        wrong = np.flatnonzero(y_pred != y_true)
        cases = [{"index": int(i), "true": CLASS_NAMES[y_true[i]],
                  "pred": CLASS_NAMES[y_pred[i]],
                  "min_confidence": float(probs[i, y_pred[i]])} for i in wrong]
        cases.sort(key=lambda r: -r["min_confidence"])
        print("\n공통 오분류 파일이 없어 이 모델의 오분류에서 고릅니다")

    wanted = [("neutrophil", "immature granulocytes"),
              ("immature granulocytes", "monocyte"),
              ("immature granulocytes", "neutrophil")]
    for true_n, pred_n in wanted:
        picked = next((c for c in cases if c["true"] == true_n and c["pred"] == pred_n), None)
        if picked is None:
            print(f"  {true_n} → {pred_n}: 해당 사례 없음")
            continue
        i = picked["index"]
        key = f"case_{SLUG[true_n]}_to_{SLUG[pred_n]}"
        save(images[i], out / f"{key}.png", args.scale)
        manifest[key] = {"true": true_n, "pred": pred_n, "index": i,
                         "min_confidence": picked["min_confidence"]}
        print(f"  {true_n} → {pred_n}  #{i} 확신 {picked['min_confidence']:.3f}")

    path = out / "cells.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n색인: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
