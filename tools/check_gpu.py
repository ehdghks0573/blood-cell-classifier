"""GPU 실동작 검증.

torch.cuda.is_available() 만으로는 부족하다. GTX 1050 Ti 는 Pascal(sm_61)
세대라 최신 PyTorch 가 커널을 안 담고 있을 수 있는데, 그 경우에도
is_available() 은 True 를 반환하고 실제 연산에서 터진다.
그래서 행렬곱과 conv 를 실제로 돌려보고, 속도까지 잰다.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.console import make_output_encodable  # noqa: E402


def main() -> int:
    make_output_encodable()
    print(f"torch          : {torch.__version__}")
    print(f"CUDA 빌드      : {torch.version.cuda}")
    print(f"is_available   : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nGPU 를 못 씁니다. CPU 로 학습해야 합니다.")
        return 1

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f"장치           : {name}")
    print(f"compute cap    : sm_{cap[0]}{cap[1]}")
    print(f"VRAM           : {vram:.1f} GB")
    print(f"지원 아키텍처  : {torch.cuda.get_arch_list()}")

    print("\n--- 실제 연산 ---")
    try:
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        gpu_s = (time.perf_counter() - t0) / 20
        print(f"행렬곱 2048^2  : {gpu_s * 1000:.1f} ms/회")
    except Exception as e:
        print(f"행렬곱 실패    : {type(e).__name__}: {e}")
        return 2

    # 전이학습에서 실제로 쓰는 경로 — conv + backward 까지 확인한다
    try:
        import torchvision

        model = torchvision.models.resnet18(weights=None, num_classes=8).cuda()
        x = torch.randn(32, 3, 224, 224, device="cuda")
        y = torch.randint(0, 8, (32,), device="cuda")
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(3):  # 워밍업
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        torch.cuda.synchronize()
        step_s = (time.perf_counter() - t0) / 10

        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"ResNet18 학습  : {step_s * 1000:.0f} ms/스텝 (배치 32, 224px)")
        print(f"VRAM 최대 사용 : {peak:.2f} GB")

        # BloodMNIST 학습 시간 어림
        steps_per_epoch = 11959 // 32
        print(f"\n예상 학습 시간 : 1 에폭 약 {steps_per_epoch * step_s:.0f}초, "
              f"20 에폭 약 {steps_per_epoch * step_s * 20 / 60:.0f}분")
    except Exception as e:
        print(f"ResNet18 실패  : {type(e).__name__}: {e}")
        return 3

    print("\nGPU 정상 동작.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
