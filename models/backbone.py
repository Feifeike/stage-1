from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    already_present = path_str in sys.path
    if not already_present:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if not already_present:
            sys.path.remove(path_str)


def center_pad_to_patch(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, height, width = images.shape
    pad_h = (-height) % patch_size
    pad_w = (-width) % patch_size
    if pad_h == 0 and pad_w == 0:
        return images
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(images, (left, right, top, bottom))


def default_dinov3_layers(depth: int) -> list[int]:
    return [depth // 4 - 1, depth // 2 - 1, depth // 4 * 3 - 1, depth - 1]


def resolve_dinov3_dim(model_name: str, output_type: str, num_layers: int) -> int:
    dims = {
        "dinov3_vits16": 384,
        "dinov3_vits16plus": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vitl16plus": 1024,
    }
    if model_name not in dims:
        raise ValueError(f"Unsupported DINOv3 model_name: {model_name}")
    per_layer = dims[model_name] * (2 if output_type == "dense-cls" else 1)
    return per_layer * num_layers


class ConvBackbone(nn.Module):
    """Small trainable fallback backbone for smoke tests and CPU debugging."""

    def __init__(self, descriptor_dim: int = 128, stride: int = 8) -> None:
        super().__init__()
        if stride not in {4, 8, 16}:
            raise ValueError("ConvBackbone stride must be 4, 8, or 16.")
        layers: list[nn.Module] = [
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        ]
        current_stride = 4
        channels = 64
        while current_stride < stride:
            layers.extend(
                [
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                    nn.GroupNorm(8, channels),
                    nn.GELU(),
                ]
            )
            current_stride *= 2
        layers.extend([nn.Conv2d(channels, descriptor_dim, 1), nn.GroupNorm(8, descriptor_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class DinoV3DenseBackbone(nn.Module):
    """Minimal standalone DINOv3 dense feature extractor.

    It loads the local DINOv3 implementation and weights directly. The output is a
    normalized `[B, descriptor_dim, H/patch, W/patch]` descriptor map.
    """

    def __init__(
        self,
        descriptor_dim: int,
        model_name: str,
        output_type: str,
        repo_path: str | Path,
        weights_path: str | Path,
        freeze_backbone: bool = True,
        layer_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        repo_path = Path(repo_path)
        weights_path = Path(weights_path)
        if not repo_path.exists():
            raise FileNotFoundError(repo_path)
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        self.model_name = model_name
        self.output_type = output_type
        self.freeze_backbone = bool(freeze_backbone)
        with prepend_sys_path(repo_path):
            backbones = importlib.import_module("dinov3.hub.backbones")
            builder = getattr(backbones, model_name)
            self.model = builder(weights=str(weights_path))
        self.patch_size = int(self.model.patch_size)
        self.layer_indices = layer_indices or default_dinov3_layers(len(self.model.blocks))
        input_dim = resolve_dinov3_dim(model_name, output_type, len(self.layer_indices))
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, descriptor_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, descriptor_dim),
            nn.GELU(),
            nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=1),
        )
        if self.freeze_backbone:
            self.model.requires_grad_(False)
            self.model.eval()

    def _tokens_to_map(self, dense_tokens: torch.Tensor, cls_token: torch.Tensor | None, feat_hw: tuple[int, int]) -> torch.Tensor:
        feat_h, feat_w = feat_hw
        dense = dense_tokens.transpose(1, 2).reshape(dense_tokens.shape[0], dense_tokens.shape[-1], feat_h, feat_w)
        if self.output_type == "dense":
            return dense
        if self.output_type == "dense-cls":
            if cls_token is None:
                raise ValueError("DINOv3 cls token is required for output_type='dense-cls'.")
            cls = cls_token[:, :, None, None].expand(-1, -1, feat_h, feat_w)
            return torch.cat((dense, cls), dim=1)
        raise ValueError(f"Unsupported DINOv3 output_type: {self.output_type}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        padded = center_pad_to_patch(image, self.patch_size)
        feat_h = padded.shape[-2] // self.patch_size
        feat_w = padded.shape[-1] // self.patch_size
        grad_enabled = not self.freeze_backbone
        with torch.set_grad_enabled(grad_enabled):
            if self.freeze_backbone:
                self.model.eval()
            outputs = self.model.get_intermediate_layers(
                padded,
                n=self.layer_indices,
                reshape=False,
                return_class_token=True,
            )
        maps = [self._tokens_to_map(tokens, cls, (feat_h, feat_w)) for tokens, cls in outputs]
        descriptors = self.proj(torch.cat(maps, dim=1))
        return F.normalize(descriptors, dim=1)


class Stage1Backbone(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.backbone_type = str(model_cfg.get("backbone_type", "dinov3"))
        self.shared = bool(model_cfg.get("shared_backbone", True))
        descriptor_dim = int(model_cfg.get("descriptor_dim", 128))
        if self.backbone_type == "conv":
            stride = int(model_cfg.get("conv_stride", 8))
            self.front = ConvBackbone(descriptor_dim=descriptor_dim, stride=stride)
            self.satellite = self.front if self.shared else ConvBackbone(descriptor_dim=descriptor_dim, stride=stride)
        elif self.backbone_type == "dinov3":
            layer_indices = model_cfg.get("layer_indices")
            front = DinoV3DenseBackbone(
                descriptor_dim=descriptor_dim,
                model_name=model_cfg.get("backbone_name", "dinov3_vits16"),
                output_type=model_cfg.get("output_type", "dense-cls"),
                repo_path=model_cfg.get("dinov3_repo_path"),
                weights_path=model_cfg.get("dinov3_weights_path"),
                freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
                layer_indices=layer_indices,
            )
            self.front = front
            self.satellite = front if self.shared else DinoV3DenseBackbone(
                descriptor_dim=descriptor_dim,
                model_name=model_cfg.get("backbone_name", "dinov3_vits16"),
                output_type=model_cfg.get("output_type", "dense-cls"),
                repo_path=model_cfg.get("dinov3_repo_path"),
                weights_path=model_cfg.get("dinov3_weights_path"),
                freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
                layer_indices=layer_indices,
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def forward(self, front_rgb: torch.Tensor, satellite_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.front(front_rgb), self.satellite(satellite_rgb)

