"""명령줄 도구들의 공통 진입 처리 (Day 12).

트레이스백은 **개발자에게 하는 말**이다. 데이터를 아직 안 받았거나 학습을
안 돌린 사람에게 40줄짜리 스택을 보여 주면, 무엇을 잘못했는지가 아니라
"프로그램이 고장 났다"는 인상만 남는다.

여기서 하는 일은 두 가지다.

  1. 사용자가 고칠 수 있는 실패를 `UserError` 로 모아 **한 줄로** 보여 준다
  2. 체크포인트 불러오기처럼 도구마다 똑같이 쓰던 코드를 한곳에 둔다

버그는 계속 트레이스백으로 터뜨린다. 감추면 못 고친다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from .console import make_output_encodable
from .data import NUM_CLASSES
from .model import build_model


class UserError(Exception):
    """사용자가 고칠 수 있는 실패. 트레이스백 없이 메시지만 보여 준다.

    "파일이 없다", "먼저 학습을 돌려야 한다" 같은 것들이다.
    코드의 버그는 여기 해당하지 않는다 — 그건 그대로 터져야 고친다.
    """


def positive_int(text: str) -> int:
    """argparse 용. 0 이나 음수를 미리 막는다.

    `--epochs 0` 은 argparse 를 통과한 뒤 학습 루프가 그냥 아무것도 안 하고
    끝나서, 사용자는 왜 결과가 없는지 모른다.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수가 필요합니다: {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다: {value}")
    return value


def pick_device(force_cpu: bool = False) -> torch.device:
    """GPU 가 있으면 GPU, 없으면 CPU. 무엇을 골랐는지 감추지 않는다."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def describe_device(device: torch.device) -> str:
    if device.type == "cpu":
        return "CPU (GPU 를 못 찾았거나 --cpu 를 줬습니다. 학습은 몇 시간 걸립니다)"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} · VRAM {vram:.1f}GB"


def load_run(run: str, root: Path, device: torch.device):
    """`runs/<run>/best.pt` 를 불러 모델과 체크포인트를 돌려준다.

    도구 다섯 개가 같은 코드를 갖고 있었고, 그중 셋은 파일이 없을 때
    트레이스백을 뱉었다. 한곳으로 모으면서 안내도 하나로 맞춘다.
    """
    run_dir = root / "runs" / run
    ckpt_path = run_dir / "best.pt"

    if not ckpt_path.exists():
        available = sorted(p.parent.name for p in (root / "runs").glob("*/best.pt"))
        hint = (f"\n지금 있는 실행: {', '.join(available)}" if available
                else "\n아직 학습한 실행이 없습니다. 먼저: python train.py --epochs 20")
        raise UserError(f"{run_dir.name} 에 학습 결과(best.pt)가 없습니다.{hint}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:  # 손상된 파일, 중간에 끊긴 저장 등
        raise UserError(
            f"{ckpt_path} 를 읽지 못했습니다 ({type(exc).__name__}).\n"
            "학습이 도중에 끊겼을 수 있습니다. 다시 학습하세요."
        ) from exc

    for key in ("arch", "model", "input_size"):
        if key in ckpt:
            continue
        raise UserError(
            f"{ckpt_path} 에 '{key}' 가 없습니다. 이 프로젝트가 만든 파일이 아닌 것 같습니다."
        )

    model = build_model(ckpt["arch"], NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


#: VRAM 이 모자랄 때 알려 줄 것. 4GB 카드에서 자주 만난다.
_OOM_HELP = """GPU 메모리가 모자랍니다.

다음 중 하나를 시도하세요 — 위에서부터 효과가 큽니다.

  --batch-size 16      배치를 줄인다 (기본 32)
  --input-size 56      입력 해상도를 낮춘다 (기본 112)
  --arch resnet18      더 작은 모델을 쓴다
  --cpu                GPU 를 쓰지 않는다 (느립니다)

다른 프로그램이 GPU 를 쓰고 있는지도 확인하세요: nvidia-smi"""


def guard(main: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    """도구의 `main` 을 감싸 실패를 사람이 읽을 수 있게 바꾼다.

    돌려주는 값이 그대로 종료 코드가 되므로, 스크립트에서 이어 붙일 때
    실패를 감지할 수 있다 (Day 11 의 run_all).
    """
    make_output_encodable()
    try:
        return main(argv)
    except UserError as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 130
    except torch.cuda.OutOfMemoryError:
        print(f"\n{_OOM_HELP}")
        return 1
    except RuntimeError as exc:
        # CUDA OOM 이 RuntimeError 로 올라오는 경우가 있다 (구버전 torch)
        if "out of memory" in str(exc).lower():
            print(f"\n{_OOM_HELP}")
            return 1
        raise
    except FileNotFoundError as exc:
        print(f"\n파일을 찾지 못했습니다: {exc.filename or exc}")
        return 1
