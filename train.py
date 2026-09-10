"""학습 진입점.

    python train.py                          # 28px 데이터, 224px 입력, ResNet18
    python train.py --size 224 --epochs 30   # 고해상도 데이터가 있을 때
    python train.py --arch resnet50 --batch-size 16
    python train.py --class-weights          # 불균형 보정 실험

실행할 때마다 runs/<이름>/ 에 설정·지표·곡선·체크포인트가 남는다.
기록 없이 값만 바꿔가며 돌리면 사흘 뒤에 뭐가 좋았는지 알 수 없게 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src import metrics as M
from src import plots
from src.cli import describe_device, guard, pick_device, positive_int
from src.data import CLASS_NAMES, NUM_CLASSES, available_sizes, build_loaders
from src.engine import class_weights, predict, train_one_epoch
from src.model import ARCHS, build_model, count_parameters
from src.seeding import seed_everything

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="백혈구 8종 분류 학습")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224],
                   help="내려받은 데이터 해상도")
    p.add_argument("--input-size", type=int, default=224,
                   help="모델에 넣을 크기. 사전학습 모델 기준이라 224 가 기본")
    p.add_argument("--arch", default="resnet18", choices=list(ARCHS))
    p.add_argument("--epochs", type=positive_int, default=20)
    p.add_argument("--batch-size", type=positive_int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weights", action="store_true",
                   help="클래스 불균형을 손실 가중치로 보정한다")
    p.add_argument("--no-pretrained", action="store_true",
                   help="사전학습 가중치 없이 밑바닥부터 학습 (비교 실험용)")
    p.add_argument("--patience", type=positive_int, default=7,
                   help="검증 macro F1 이 이만큼 개선 없으면 조기 종료")
    p.add_argument("--name", help="실행 이름. 없으면 시각으로 자동 생성")
    p.add_argument("--cpu", action="store_true", help="GPU 가 있어도 CPU 로 학습")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    have = available_sizes(args.data_root)
    if args.size not in have:
        print(f"{args.size}px 데이터가 없습니다. 사용 가능: {have if have else '없음'}")
        print(f"tools/fetch_data.py 를 실행하거나 npz 를 {args.data_root} 에 넣으세요.")
        return 1

    device = pick_device(args.cpu)
    print(f"장치     : {describe_device(device)}")
    seed_everything(args.seed)

    name = args.name or f"{args.arch}_{args.size}px_{datetime.now():%m%d_%H%M%S}"
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(
        args.data_root, size=args.size, input_size=args.input_size,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )

    model = build_model(args.arch, NUM_CLASSES, pretrained=not args.no_pretrained).to(device)
    total, trainable = count_parameters(model)

    weights = class_weights(loaders.class_counts, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"실행     : {name}")
    print(f"장치     : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"모델     : {args.arch}  파라미터 {total / 1e6:.1f}M")
    print(f"데이터   : {args.size}px → 입력 {args.input_size}px  "
          f"train {len(loaders.train.dataset):,} / val {len(loaders.val.dataset):,} "
          f"/ test {len(loaders.test.dataset):,}")
    print(f"손실     : CrossEntropy" + (" (클래스 가중치 적용)" if args.class_weights else ""))
    print()

    history: list[dict] = []
    best_f1, best_epoch, since_best = -1.0, 0, 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders.train, criterion, optimizer,
                                     device, epoch, args.epochs)
        y_true, y_pred, _, val_loss = predict(model, loaders.val, device, criterion)
        m = M.compute(y_true, y_pred, NUM_CLASSES)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": m.accuracy,
            "val_macro_f1": m.macro_f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        marker = ""
        if m.macro_f1 > best_f1:
            best_f1, best_epoch, since_best = m.macro_f1, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "arch": args.arch,
                "input_size": args.input_size,
                "epoch": epoch,
                "val_macro_f1": m.macro_f1,
            }, run_dir / "best.pt")
            marker = "  ← 최고"
        else:
            since_best += 1

        print(f"  {epoch:3d}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"acc {m.accuracy:.4f}  macroF1 {m.macro_f1:.4f}{marker}")

        if since_best >= args.patience:
            print(f"\n  {args.patience} 에폭 동안 개선이 없어 조기 종료합니다.")
            break

    elapsed = time.perf_counter() - started
    print(f"\n학습 완료: {elapsed / 60:.1f}분, 최고 val macro F1 {best_f1:.4f} (에폭 {best_epoch})")

    # 최고 성능 지점으로 되돌린 뒤 test 로 최종 평가
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    y_true, y_pred, _, _ = predict(model, loaders.test, device)
    test_m = M.compute(y_true, y_pred, NUM_CLASSES)

    worst_i, worst_f1 = test_m.worst_class
    print(f"\n=== test ===")
    print(f"accuracy    {test_m.accuracy:.4f}")
    print(f"macro F1    {test_m.macro_f1:.4f}   {'✅' if test_m.macro_f1 >= 0.90 else '❌'} S1 (≥0.90)")
    print(f"최저 클래스 {CLASS_NAMES[worst_i]} {worst_f1:.4f}   "
          f"{'✅' if worst_f1 >= 0.80 else '❌'} S2 (≥0.80)")
    print(f"학습 시간   {elapsed / 60:.1f}분   {'✅' if elapsed < 1800 else '❌'} S6 (<30분)")
    print()
    print(M.report_text(y_true, y_pred, CLASS_NAMES))

    print("가장 많이 헷갈린 쌍:")
    for c in M.top_confusions(test_m.confusion, CLASS_NAMES, k=3):
        print(f"  {c['true']} → {c['pred']}  {c['count']}장 ({c['rate']:.1%})")

    result = {
        "name": name,
        "args": vars(args),
        "device": str(device),
        "params_total": total,
        "params_trainable": trainable,
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test": test_m.to_dict(),
        "history": history,
        "class_names": CLASS_NAMES,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    plots.training_curves(history, run_dir / "curves.png")
    plots.confusion_heatmap(test_m.confusion, CLASS_NAMES, run_dir / "confusion.png")

    print(f"\n저장: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(guard(main))
