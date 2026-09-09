"""재현성.

성공 기준 S3 이 "같은 시드로 두 번 돌려 macro F1 차이 0.01 미만"이다.
시드를 안 고정하면 실험 비교 자체가 무의미해진다 — 값을 바꿔서 좋아진 건지
그냥 운이 좋았던 건지 구분할 수 없다.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """파이썬, numpy, torch 의 난수를 모두 고정한다.

    deterministic=True 면 cuDNN 이 매번 같은 알고리즘을 쓰게 한다.
    조금 느려지지만 결과가 재현된다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 워커마다 다른, 그러나 재현되는 시드를 준다."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
