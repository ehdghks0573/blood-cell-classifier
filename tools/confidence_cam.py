"""확신이 높을 때와 낮을 때, 모델은 다른 곳을 보는가 (Day 4).

    python tools/confidence_cam.py --run effb0_112

`explain.py` 는 맞힌 장과 틀린 장을 나란히 놓는다. 여기서는 **맞힌 장 안에서도**
확신의 높낮이로 나눈다. 묶음은 셋이다.

  - 확신 높은 정답 — 예측 확률 ≥ 0.99
  - 확신 낮은 정답 — 예측 확률 < 0.90
  - 오답

**물음.** 확신이 낮은 이유가 "엉뚱한 곳을 봐서"라면, 확신 낮은 정답과 오답에서
히트맵이 세포 밖으로 새야 한다. 새지 않는다면 망설임은 보는 위치가 아니라
**본 것의 내용**에서 온다 — 5절의 "모델은 세포를 제대로 보는데도 틀린다"와
같은 이야기가 된다.

한 장마다 숫자는 `src/attention.py` 의 "세포 위 비율"이다. 대조군은 둘이다 —
같은 히트맵에 ① 다른 장의 마스크, ② 면적이 같은 정중앙 원을 댄 값. 세포가
늘 가운데 있는 데이터라 ① 만으로는 "가운데를 본다"와 갈리지 않았다.
판정은 부트스트랩 95% 구간으로 하고, 구간이 0 을 걸치면 "판정보류"다.

결과: `runs/<run>/confidence_cam.json`, `runs/<run>/gradcam_confidence.png`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import attention as A  # noqa: E402
from src.cli import guard, load_run, pick_device  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_NAMES,
    BloodDataset,
    build_transform,
    denormalize,
    load_split,
)
from src.gradcam import GradCAM, overlay  # noqa: E402
from src.model import finer_layer, target_layer  # noqa: E402
from src.plots import short_names, use_korean_font  # noqa: E402

HIGH = 0.99
LOW = 0.90

GROUPS = {
    "high": f"확신 높은 정답 (≥ {HIGH:.2f})",
    "low": f"확신 낮은 정답 (< {LOW:.2f})",
    "wrong": "오답",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="확신도별 Grad-CAM 비교")
    p.add_argument("--run", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--show", type=int, default=6, help="그림에 묶음마다 보일 장수")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", default="last", choices=["last", "finer"],
                   help="finer: 해상도가 두 배인 앞 블록 (112px 입력에서 7×7)")
    p.add_argument("--mask", default="otsu", choices=["otsu", "largest"],
                   help="largest: 가장 큰 어두운 덩어리만 세포로 본다")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


def assign_groups(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """장마다 'high' / 'low' / 'wrong' / '' (어디에도 안 속함)."""
    pred = probs.argmax(axis=1)
    p = probs.max(axis=1)
    out = np.full(len(y_true), "", dtype=object)
    right = pred == y_true
    out[right & (p >= HIGH)] = "high"
    out[right & (p < LOW)] = "low"
    out[~right] = "wrong"
    return out


def summarize(own: dict[str, np.ndarray], swapped: dict[str, np.ndarray],
              disk: dict[str, np.ndarray]) -> dict:
    """묶음별 평균과 구간, 두 대조군과의 차이, 묶음 사이 차이."""
    report: dict = {"groups": {}, "comparisons": {}}
    for g in GROUPS:
        if len(own[g]) == 0:
            continue
        m, lo, hi = A.mean_ci(own[g])
        cm, clo, chi = A.mean_ci(swapped[g])
        d, dlo, dhi = A.diff_ci(own[g], swapped[g])
        km, klo, khi = A.mean_ci(disk[g])
        e, elo, ehi = A.diff_ci(own[g], disk[g])
        report["groups"][g] = {
            "label": GROUPS[g], "n": int(len(own[g])),
            "inside": [round(m, 4), round(lo, 4), round(hi, 4)],
            "control": [round(cm, 4), round(clo, 4), round(chi, 4)],
            "own_minus_control": [round(d, 4), round(dlo, 4), round(dhi, 4)],
            "control_verdict": A.verdict(dlo, dhi),
            "disk": [round(km, 4), round(klo, 4), round(khi, 4)],
            "own_minus_disk": [round(e, 4), round(elo, 4), round(ehi, 4)],
            "disk_verdict": A.verdict(elo, ehi),
        }
    for a, b in (("high", "low"), ("high", "wrong"), ("low", "wrong")):
        if len(own[a]) and len(own[b]):
            d, lo, hi = A.diff_ci(own[a], own[b])
            report["comparisons"][f"{a}-{b}"] = {
                "diff": [round(d, 4), round(lo, 4), round(hi, 4)],
                "verdict": A.verdict(lo, hi),
            }
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_korean_font()
    device = pick_device(args.cpu)
    model, ckpt = load_run(args.run, ROOT, device)
    pick_layer = target_layer if args.layer == "last" else finer_layer
    layer = pick_layer(model, ckpt["arch"])
    largest = args.mask == "largest"
    run_dir = ROOT / "runs" / args.run
    tag = "" if (args.layer, args.mask) == ("last", "otsu") else f"_{args.layer}_{args.mask}"

    imgs, lbls = load_split(args.data_root, args.size, args.split)
    labels = np.asarray(lbls).reshape(-1)
    size = ckpt["input_size"]
    dataset = BloodDataset(imgs, lbls, build_transform(size, train=False))
    print(f"실행 {args.run} · {ckpt['arch']} · {args.split} {len(dataset):,}장")

    # 1) 확률 — 묶음을 나누는 데만 쓴다
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    probs = np.concatenate(probs)
    groups = assign_groups(labels, probs)

    # 2) 묶음에 든 장마다 예측 클래스의 히트맵과 세포 마스크
    rng = np.random.default_rng(args.seed)
    own = {g: [] for g in GROUPS}
    swapped = {g: [] for g in GROUPS}
    disk = {g: [] for g in GROUPS}
    shown: dict[str, list] = {g: [] for g in GROUPS}
    members = {g: np.flatnonzero(groups == g) for g in GROUPS}
    picks = {g: set(rng.choice(m, size=min(args.show, len(m)), replace=False).tolist())
             for g, m in members.items()}

    with GradCAM(model, layer) as cam:
        for g, idx in members.items():
            heats, masks = [], []
            for i in idx:
                image, _ = dataset[int(i)]
                heat, pred, p = cam(image.unsqueeze(0).to(device))
                mask = A.cell_mask(imgs[i], size, largest)
                heats.append(heat)
                masks.append(mask)
                own[g].append(A.inside_ratio(heat, mask))
                disk[g].append(A.inside_ratio(heat, A.centered_disk(mask)))
                if int(i) in picks[g]:
                    shown[g].append((int(i), image, heat, mask, pred, float(p[pred])))
            # 대조군: 같은 묶음 안에서 마스크를 한 칸씩 밀어 남의 마스크를 댄다
            for k, heat in enumerate(heats):
                swapped[g].append(A.inside_ratio(heat, masks[(k + 1) % len(masks)]))
            print(f"  {GROUPS[g]:22s} {len(idx):5d}장")

    own = {g: np.asarray(v) for g, v in own.items()}
    swapped = {g: np.asarray(v) for g, v in swapped.items()}
    disk = {g: np.asarray(v) for g, v in disk.items()}
    report = summarize(own, swapped, disk)
    report.update({"run": args.run, "split": args.split, "high": HIGH, "low": LOW,
                   "layer": args.layer, "mask": args.mask,
                   "mask_area": round(float(np.mean([A.cell_mask(im, size, largest).mean()
                                                      for im in imgs[:500]])), 4)})

    print(f"\n세포 마스크 평균 면적 {report['mask_area']:.3f} (고른 히트맵이면 이 값)")
    print("세포 위 비율 (평균 · 95% 구간) │ ① 남의 마스크 │ ② 같은 면적의 가운데 원")
    for g, r in report["groups"].items():
        m, lo, hi = r["inside"]
        e, elo, ehi = r["own_minus_disk"]
        print(f"  {r['label']:22s} {m:.3f} ({lo:.3f}~{hi:.3f}) │ "
              f"① {r['control'][0]:.3f} 제 것이 {r['control_verdict']} │ "
              f"② {r['disk'][0]:.3f} 차이 {e:+.3f} ({elo:+.3f}~{ehi:+.3f}) 제 것이 {r['disk_verdict']}")
    print("\n묶음 사이 차이")
    for k, r in report["comparisons"].items():
        d, lo, hi = r["diff"]
        print(f"  {k:11s} {d:+.3f} ({lo:+.3f}~{hi:+.3f})  → {r['verdict']}")

    out = run_dir / f"confidence_cam{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {out}")
    draw(shown, report, run_dir / f"gradcam_confidence{tag}.png")
    return 0


def draw(shown: dict, report: dict, path: Path) -> None:
    rows = [g for g in GROUPS if shown[g]]
    cols = max(len(shown[g]) for g in rows)
    fig, axes = plt.subplots(len(rows), cols * 2, figsize=(cols * 2 * 1.25, len(rows) * 1.7),
                             squeeze=False)
    short = short_names(CLASS_NAMES)
    for r, g in enumerate(rows):
        for c in range(cols):
            ax_img, ax_cam = axes[r, c * 2], axes[r, c * 2 + 1]
            for ax in (ax_img, ax_cam):
                ax.set_xticks([])
                ax.set_yticks([])
            if c >= len(shown[g]):
                ax_img.axis("off")
                ax_cam.axis("off")
                continue
            i, image, heat, mask, pred, p = shown[g][c]
            base = denormalize(image).cpu().numpy()
            ax_img.imshow(np.transpose(base, (1, 2, 0)))
            ax_img.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
            ax_cam.imshow(overlay(base, heat))
            ax_cam.set_title(f"{short[pred]} {p:.2f}", fontsize=6, pad=1)
            ax_img.set_title(f"#{i}", fontsize=6, pad=1)
        stats = report["groups"][g]
        axes[r, 0].set_ylabel(f"{stats['label']}\n세포 위 {stats['inside'][0]:.2f}",
                              rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("확신도별 Grad-CAM — 각 쌍은 (원본 + 세포 윤곽, 히트맵) · 제목은 예측과 확률",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    sys.exit(guard(main))
