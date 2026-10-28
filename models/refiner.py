from __future__ import annotations

"""RoMa-style fine refinement for the sparse Stage-1 query grid.

The original implementation used one MLP per query. RoMa instead refines a
normalized warp with a ConvRefiner at several image scales. This file keeps
the same idea while retaining the current project's sparse front-view query
grid: feature maps from the front image are sampled at the fixed query
locations, and the ConvRefiner operates on that regular query grid.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_uv_to_normalized(uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Convert pixel coordinates to RoMa/grid_sample coordinates."""

    image_h, image_w = image_hw
    x = (uv[..., 0] + 0.5) / max(float(image_w), 1.0) * 2.0 - 1.0
    y = (uv[..., 1] + 0.5) / max(float(image_h), 1.0) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)


def normalized_to_pixel_uv(coords: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Convert ``align_corners=False`` normalized coordinates to pixels."""

    image_h, image_w = image_hw
    x = (coords[..., 0] + 1.0) * max(float(image_w), 1.0) * 0.5 - 0.5
    y = (coords[..., 1] + 1.0) * max(float(image_h), 1.0) * 0.5 - 0.5
    return torch.stack((x, y), dim=-1)


def sample_map_at_uv(
    feature_map: torch.Tensor,
    uv: torch.Tensor,
    image_hw: tuple[int, int],
    *,
    align_corners: bool = False,
) -> torch.Tensor:
    """Bilinearly sample ``[B,C,H,W]`` at image pixel coordinates.

    Returns ``[B,N,C]``. ``align_corners=False`` is RoMa's convention; the
    keyword remains configurable for compatibility with old callers.
    """

    if align_corners:
        image_h, image_w = image_hw
        x = uv[..., 0] / max(image_w - 1, 1) * 2.0 - 1.0
        y = uv[..., 1] / max(image_h - 1, 1) * 2.0 - 1.0
        normalized = torch.stack((x, y), dim=-1)
    else:
        normalized = pixel_uv_to_normalized(uv, image_hw)
    grid = normalized.unsqueeze(2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous()


def _local_correlation(
    source_feature: torch.Tensor,
    target_feature: torch.Tensor,
    warp: torch.Tensor,
    radius: int,
    *,
    disable_grad: bool = True,
) -> torch.Tensor:
    """RoMa's native local correlation, generalized to unequal image sizes."""

    batch_size, channels, query_h, query_w = source_feature.shape
    target_h, target_w = target_feature.shape[-2:]
    offsets = torch.arange(-radius, radius + 1, device=warp.device, dtype=warp.dtype)
    oy, ox = torch.meshgrid(
        offsets * (2.0 / max(float(target_h), 1.0)),
        offsets * (2.0 / max(float(target_w), 1.0)),
        indexing="ij",
    )
    local_offsets = torch.stack((ox.reshape(-1), oy.reshape(-1)), dim=-1)
    window = warp.detach().permute(0, 2, 3, 1)[:, :, :, None, :] + local_offsets[None, None, None]
    window = window.reshape(batch_size, query_h, query_w * local_offsets.shape[0], 2)
    grad_context = torch.no_grad() if disable_grad else torch.enable_grad()
    with grad_context:
        sampled = F.grid_sample(
            target_feature,
            window,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    window_size = local_offsets.shape[0]
    sampled = sampled.reshape(batch_size, channels, query_h, query_w, window_size)
    correlation = (source_feature[:, :, :, :, None] * sampled / (float(channels) ** 0.5)).sum(dim=1)
    return correlation.permute(0, 3, 1, 2).contiguous()


class _VGGBlock(nn.Module):
    """A dependency-free VGG19-BN block used by RoMa's fine encoder."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_convs):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RomaFineFeaturePyramid(nn.Module):
    """VGG19-BN-shaped fine feature extractor at scales 1, 2, 4 and 8.

    Official RoMa initializes this encoder from ImageNet-pretrained VGG19-BN.
    The current environment does not provide torchvision, so the same block
    structure is implemented locally and trained with the refinement stage.
    """

    channels = {1: 64, 2: 128, 4: 256, 8: 512}

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _VGGBlock(3, 64, 2),
                _VGGBlock(64, 128, 2),
                _VGGBlock(128, 256, 4),
                _VGGBlock(256, 512, 4),
            ]
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained:
            self._load_torchvision_vgg19_bn()

    def _load_torchvision_vgg19_bn(self) -> None:
        """Optionally initialize from the official ImageNet VGG19-BN weights."""

        try:
            import torchvision.models as tvm
        except ImportError as exc:
            raise RuntimeError(
                "refiner.fine_pretrained=true requires torchvision; set it to false "
                "or install torchvision in the training environment."
            ) from exc
        weights = tvm.vgg.VGG19_BN_Weights.IMAGENET1K_V1
        pretrained_model = tvm.vgg19_bn(weights=weights)
        source_blocks: list[list[nn.Module]] = []
        current: list[nn.Module] = []
        for layer in pretrained_model.features:
            if isinstance(layer, nn.MaxPool2d):
                source_blocks.append(current)
                current = []
                if len(source_blocks) == len(self.blocks):
                    break
            else:
                current.append(layer)
        if len(source_blocks) != len(self.blocks):
            raise RuntimeError("Could not split torchvision VGG19-BN into four feature blocks.")
        for target, source in zip(self.blocks, source_blocks):
            target.net.load_state_dict(nn.Sequential(*source).state_dict())

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        x = (image - self.mean.to(image)) / self.std.to(image)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = block(x)
            scale = 2**index
            features[scale] = x
            if index + 1 < len(self.blocks):
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return features


class RomaConvRefiner(nn.Module):
    """The ConvRefiner used by RoMa, adapted to sparse source queries."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        displacement_emb_dim: int | None,
        local_corr_radius: int | None,
        hidden_blocks: int = 8,
        depthwise: bool = True,
        kernel_size: int = 5,
        bn_momentum: float = 0.01,
        disable_local_corr_grad: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.local_corr_radius = int(local_corr_radius or 0)
        self.disable_local_corr_grad = bool(disable_local_corr_grad)
        self.has_displacement_emb = displacement_emb_dim is not None
        if self.has_displacement_emb:
            self.disp_emb = nn.Conv2d(2, int(displacement_emb_dim), kernel_size=1)

        def create_block(input_dim: int, output_dim: int, *, first: bool = False) -> nn.Sequential:
            # RoMa uses depthwise convolution. A custom hidden width that is
            # not divisible by the input width falls back to a dense block.
            groups = input_dim if depthwise and output_dim % input_dim == 0 else 1
            return nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=groups,
                    bias=first,
                ),
                nn.BatchNorm2d(output_dim, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_dim, output_dim, kernel_size=1),
            )

        self.block1 = create_block(self.in_dim, int(hidden_dim), first=True)
        self.hidden_blocks = nn.Sequential(
            *[create_block(int(hidden_dim), int(hidden_dim)) for _ in range(int(hidden_blocks))]
        )
        self.out_conv = nn.Conv2d(int(hidden_dim), 3, kernel_size=1)

    def forward(
        self,
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        warp: torch.Tensor,
        source_coords: torch.Tensor,
        *,
        scale_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_at_warp = F.grid_sample(
            target_feature,
            warp.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        pieces = [source_feature, target_at_warp]
        if self.has_displacement_emb:
            displacement = warp - source_coords
            pieces.append(self.disp_emb((40.0 / 32.0) * scale_factor * displacement))
        if self.local_corr_radius > 0:
            pieces.append(
                _local_correlation(
                    source_feature,
                    target_feature,
                    warp,
                    self.local_corr_radius,
                    disable_grad=self.disable_local_corr_grad,
                )
            )
        features = torch.cat(pieces, dim=1)
        if features.shape[1] < self.in_dim:
            features = F.pad(features, (0, 0, 0, 0, 0, self.in_dim - features.shape[1]))
        elif features.shape[1] > self.in_dim:
            raise RuntimeError(
                f"RoMa ConvRefiner input has {features.shape[1]} channels, expected {self.in_dim}."
            )
        features = self.block1(features)
        features = self.hidden_blocks(features)
        output = self.out_conv(features.float())
        return output[:, :2], output[:, 2:3]


class LocalRefiner(nn.Module):
    """Multi-scale RoMa-style sparse refiner.

    ``front_query_uv`` stays fixed throughout refinement. The satellite warp is
    represented in normalized coordinates and updated at 16, 8, 4, 2 and 1.
    """

    def __init__(
        self,
        dim: int,
        *,
        radius: int = 2,
        hidden_dim: int | None = None,
        max_offset_px: float = 16.0,
        scales: Iterable[int] = (16, 8, 4, 2, 1),
        fine_feature_dim: int = 64,
        fine_pretrained: bool = False,
        hidden_blocks: int = 8,
        depthwise: bool = True,
        displacement_emb_dims: dict[int, int] | None = None,
        local_corr_radii: dict[int, int | None] | None = None,
    ) -> None:
        super().__init__()
        del radius, max_offset_px  # retained as config compatibility arguments
        self.dim = int(dim)
        self.scales = tuple(int(scale) for scale in scales)
        self.fine = RomaFineFeaturePyramid(pretrained=fine_pretrained)
        self.fine_feature_dim = int(fine_feature_dim)
        self.projections = nn.ModuleDict()
        for scale in (1, 2, 4, 8):
            self.projections[str(scale)] = nn.Sequential(
                nn.Conv2d(RomaFineFeaturePyramid.channels[scale], self.fine_feature_dim, 1, bias=False),
                nn.BatchNorm2d(self.fine_feature_dim, momentum=0.01),
            )
        self.projections["16"] = nn.Identity()

        default_corr = {16: 7, 8: 3, 4: 2, 2: None, 1: None}
        default_disp = {16: 128, 8: 64, 4: 32, 2: 16, 1: 6}
        corr_radii = {**default_corr, **(local_corr_radii or {})}
        disp_dims = {**default_disp, **(displacement_emb_dims or {})}
        refiners: dict[str, RomaConvRefiner] = {}
        for scale in self.scales:
            if scale not in {1, 2, 4, 8, 16}:
                raise ValueError(f"Unsupported RoMa refinement scale: {scale}")
            feature_dim = self.dim if scale == 16 else self.fine_feature_dim
            corr_dim = (2 * int(corr_radii[scale]) + 1) ** 2 if corr_radii[scale] else 0
            input_dim = 2 * feature_dim + int(disp_dims[scale]) + corr_dim
            # Original RoMa uses hidden_dim == in_dim with depthwise blocks.
            stage_hidden = int(hidden_dim) if hidden_dim is not None else input_dim
            refiners[str(scale)] = RomaConvRefiner(
                in_dim=input_dim,
                hidden_dim=stage_hidden,
                displacement_emb_dim=int(disp_dims[scale]),
                local_corr_radius=corr_radii[scale],
                hidden_blocks=hidden_blocks,
                depthwise=depthwise,
            )
        self.refiners = nn.ModuleDict(refiners)

    def forward(
        self,
        front_features: torch.Tensor,
        satellite_feature_map: torch.Tensor,
        front_rgb: torch.Tensor,
        satellite_rgb: torch.Tensor,
        front_query_uv: torch.Tensor,
        coarse_uv: torch.Tensor,
        satellite_image_hw: tuple[int, int],
        initial_matchability_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, query_h, query_w = front_features.shape
        front_image_hw = (front_rgb.shape[-2], front_rgb.shape[-1])
        source_coords = pixel_uv_to_normalized(front_query_uv, front_image_hw)
        source_coords = source_coords.reshape(batch_size, query_h, query_w, 2).permute(0, 3, 1, 2).contiguous()
        warp = pixel_uv_to_normalized(coarse_uv, satellite_image_hw)
        warp = warp.reshape(batch_size, query_h, query_w, 2).permute(0, 3, 1, 2).contiguous()
        certainty_logits = initial_matchability_logits.reshape(batch_size, 1, query_h, query_w)

        # RoMa's CNN is Siamese and is evaluated once per image. DINO scale16
        # features come from the current coarse backbone.
        front_fine = self.fine(front_rgb)
        satellite_fine = self.fine(satellite_rgb)
        feature_maps_front: dict[int, torch.Tensor] = {16: front_features}
        feature_maps_satellite: dict[int, torch.Tensor] = {16: satellite_feature_map}
        for scale in (1, 2, 4, 8):
            feature_maps_front[scale] = self.projections[str(scale)](front_fine[scale])
            feature_maps_satellite[scale] = self.projections[str(scale)](satellite_fine[scale])

        target_h, target_w = satellite_image_hw
        for scale in self.scales:
            source_sampled = sample_map_at_uv(
                feature_maps_front[scale],
                front_query_uv,
                front_image_hw,
                align_corners=False,
            )
            source_sampled = source_sampled.transpose(1, 2).reshape(batch_size, -1, query_h, query_w)
            delta_flow, delta_certainty = self.refiners[str(scale)](
                source_sampled,
                feature_maps_satellite[scale],
                warp,
                source_coords,
                scale_factor=float(scale) / 16.0,
            )
            # RoMa predicts displacement in current feature-grid pixels and
            # converts it to normalized image coordinates before accumulation.
            warp = warp + float(scale) * torch.stack(
                (
                    delta_flow[:, 0] / (4.0 * max(float(target_w), 1.0)),
                    delta_flow[:, 1] / (4.0 * max(float(target_h), 1.0)),
                ),
                dim=1,
            )
            certainty_logits = certainty_logits + delta_certainty

        refined_norm = warp.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        refined_uv = normalized_to_pixel_uv(refined_norm, satellite_image_hw)
        return refined_uv, certainty_logits.reshape(batch_size, -1)
