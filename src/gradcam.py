"""Grad-CAM — 모델이 이미지의 어디를 보고 판단했는지 히트맵으로 보여준다.

원리: 마지막 conv 층의 특징 맵 A^k 와, 예측 점수를 그 특징 맵으로 미분한
기울기를 구한다. 기울기를 공간 방향으로 평균 내면 채널별 중요도 가중치가
되고, 그 가중치로 특징 맵을 합친 뒤 ReLU 를 씌우면 "그 클래스에 긍정적으로
기여한 영역"만 남는다.

분류만 하는 프로젝트는 흔하다. 이 프로젝트의 차별점은 모델이 세포를 보고
판단했는지, 아니면 배경 얼룩 같은 엉뚱한 단서를 보고 맞힌 것인지를
확인할 수 있다는 데 있다. 정확도 숫자만으로는 절대 알 수 없는 부분이다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """지정한 층의 활성화와 기울기를 후크로 받아 히트맵을 만든다.

        with GradCAM(model, layer) as cam:
            heatmap = cam(image_tensor, class_index)

    후크는 반드시 해제해야 한다. 안 그러면 이후 추론에서도 계속 붙어 있어
    메모리를 잡아먹는다.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = []

    def __enter__(self) -> GradCAM:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None):
        """히트맵과 예측 결과를 돌려준다.

        Args:
            image: (1, C, H, W) 정규화된 입력 한 장
            class_index: 설명할 클래스. None 이면 모델의 예측 클래스

        Returns:
            (heatmap, class_index, probabilities)
            heatmap 은 입력 크기로 늘린 0~1 사이 (H, W) 배열
        """
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError(f"(1, C, H, W) 가 필요합니다. 받은 크기: {tuple(image.shape)}")

        was_training = self.model.training
        self.model.eval()

        # 기울기가 필요하므로 no_grad 를 쓰면 안 된다.
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("후크가 걸리지 않았습니다. with 문 안에서 호출하세요.")

        # 채널별 중요도 = 기울기의 공간 평균
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # 그 클래스에 긍정적으로 기여한 영역만 남긴다

        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]

        # 0~1 로 정규화. 전부 0 이면(기여 영역 없음) 그대로 둔다.
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)

        if was_training:
            self.model.train()

        return cam.cpu().numpy(), class_index, probs


def overlay(image_chw: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """원본 이미지(0~1, CHW) 위에 히트맵을 겹쳐 RGB 배열로 돌려준다.

    matplotlib 의 jet 계열 대신 직접 만든 적-노랑 그라데이션을 쓴다.
    혈액 도말은 보라·분홍 계열이라, 파란색이 섞이는 팔레트를 쓰면
    히트맵인지 세포 염색인지 구분이 안 된다.
    """
    img = np.transpose(image_chw, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img, 0, 1)

    h = np.clip(heatmap, 0, 1)[..., None]
    # 낮음 = 검정, 중간 = 빨강, 높음 = 노랑
    warm = np.concatenate([
        np.clip(h * 2, 0, 1),
        np.clip(h * 2 - 1, 0, 1),
        np.zeros_like(h),
    ], axis=-1)

    return np.clip(img * (1 - alpha * h) + warm * (alpha * h), 0, 1)
