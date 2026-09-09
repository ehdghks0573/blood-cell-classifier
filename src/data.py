"""BloodMNIST 데이터 로딩.

medmnist 패키지의 자동 다운로드는 Zenodo 를 쓰는데 막히는 환경이 있어서,
npz 파일을 직접 읽는다. 파일이 어떤 경로로 들어왔든 있기만 하면 동작한다.

해상도(28/64/128/224)는 파일명으로 구분되고, 모델 입력 크기는 그와 별개로
정한다. 28px 데이터를 224px 로 늘려 ImageNet 사전학습 모델에 넣는 것도
가능하기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#: 8종 백혈구. medmnist INFO 와 같은 순서다.
CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

CLASS_NAMES_KO = [
    "호염기구",
    "호산구",
    "적아구",
    "미성숙 과립구",
    "림프구",
    "단핵구",
    "호중구",
    "혈소판",
]

NUM_CLASSES = len(CLASS_NAMES)

# ImageNet 사전학습 가중치를 쓰므로 그 통계로 정규화한다.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def npz_path(root: Path | str, size: int) -> Path:
    """해상도에 맞는 npz 경로. 28px 만 접미사가 없다 (medmnist 규칙)."""
    root = Path(root)
    name = "bloodmnist.npz" if size == 28 else f"bloodmnist_{size}.npz"
    return root / name


def available_sizes(root: Path | str) -> list[int]:
    """지금 내려받아 둔 해상도 목록."""
    return [s for s in (28, 64, 128, 224) if npz_path(root, s).exists()]


class BloodDataset(Dataset):
    """npz 한 분할을 감싼 Dataset.

    이미지는 uint8 HWC 로 메모리에 올려 두고, 변환은 꺼낼 때 적용한다.
    28px 는 전체가 30MB 남짓이고 224px 도 2.6GB 라 16GB RAM 에 들어간다.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        if len(images) != len(labels):
            raise ValueError(f"이미지 {len(images)}개와 라벨 {len(labels)}개가 다릅니다")
        self.images = images
        self.labels = labels.astype(np.int64).reshape(-1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=NUM_CLASSES)


def load_split(root: Path | str, size: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """npz 에서 한 분할을 꺼낸다."""
    path = npz_path(root, size)
    if not path.exists():
        have = available_sizes(root)
        raise FileNotFoundError(
            f"{path} 가 없습니다.\n"
            f"현재 사용 가능한 해상도: {have if have else '없음'}\n"
            f"tools/fetch_data.py 를 실행하거나 npz 를 {Path(root)} 에 넣으세요."
        )
    with np.load(path) as z:
        return z[f"{split}_images"], z[f"{split}_labels"]


def build_transform(input_size: int, train: bool):
    """전처리·증강 파이프라인.

    현미경 이미지는 위아래·좌우 방향에 의미가 없으므로 뒤집기와 회전이 안전하다.
    반면 색상 변형은 약하게만 준다 — 염색 색상이 세포 종류를 구분하는 실제
    단서라서, 세게 흔들면 오히려 단서를 지우게 된다.
    """
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_counts: np.ndarray


def build_loaders(
    root: Path | str,
    size: int = 28,
    input_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Loaders:
    """세 분할의 DataLoader 를 만든다."""
    from .seeding import worker_init_fn

    sets = {}
    for split in ("train", "val", "test"):
        imgs, lbls = load_split(root, size, split)
        sets[split] = BloodDataset(imgs, lbls, build_transform(input_size, split == "train"))

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    return Loaders(
        train=DataLoader(sets["train"], batch_size=batch_size, shuffle=True,
                         drop_last=True, generator=generator, **common),
        val=DataLoader(sets["val"], batch_size=batch_size * 2, shuffle=False, **common),
        test=DataLoader(sets["test"], batch_size=batch_size * 2, shuffle=False, **common),
        class_counts=sets["train"].class_counts(),
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """정규화를 되돌린다. 히트맵을 원본 위에 겹칠 때 쓴다."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
