from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class Stage1SampleRecord:
    front_rgb_path: Path
    satellite_rgb_path: Path
    # Optional auxiliary monocular depth (UniDepth/DA3). Current Stage-1 GT
    # construction uses lidar_depth instead, so this must not make a sample
    # unloadable when the auxiliary file is absent on another server.
    depth_path: Path | None
    depth_valid_mask_path: Path | None
    depth_confidence_path: Path | None
    lidar_depth_path: Path | None
    intrinsics: list[list[float]]
    camera_to_vehicle: list[list[float]]
    vehicle_to_world: list[list[float]]
    satellite_meters_per_pixel: float
    satellite_geometry_valid: bool
    satellite_center_world_xy: tuple[float, float] = (0.0, 0.0)
    satellite_forward_world_xy: tuple[float, float] | None = None
    satellite_right_world_xy: tuple[float, float] | None = None
    negative_mask_path: Path | None = None
    sample_id: str | None = None


@dataclass
class Stage1Sample:
    sample_id: str
    front_rgb: torch.Tensor
    satellite_rgb: torch.Tensor
    # Kept as an optional sample field; collate fills missing values with a
    # zero tensor so the existing batch interface remains unchanged.
    da3_depth: torch.Tensor | None
    da3_depth_valid_mask: torch.Tensor
    da3_depth_confidence: torch.Tensor
    lidar_depth: torch.Tensor | None
    intrinsics: torch.Tensor
    camera_to_vehicle: torch.Tensor
    vehicle_to_world: torch.Tensor
    camera_to_world: torch.Tensor
    satellite_meters_per_pixel: float
    satellite_geometry_valid: bool
    satellite_center_world_xy: tuple[float, float]
    satellite_forward_world_xy: tuple[float, float]
    satellite_right_world_xy: tuple[float, float]
    satellite_is_ego_oriented: bool
    explicit_negative_mask: torch.Tensor | None


@dataclass
class SampleLoadError:
    index: int
    sample_id: str
    error_type: str
    error_message: str
    front_rgb_path: str
    satellite_rgb_path: str
    depth_path: str | None
    lidar_depth_path: str | None


@dataclass
class Stage1ProjectionBatch:
    sample_ids: list[str]
    front_rgb: torch.Tensor
    satellite_rgb: torch.Tensor
    da3_depth: torch.Tensor
    da3_depth_valid_mask: torch.Tensor
    da3_depth_confidence: torch.Tensor
    lidar_depth: torch.Tensor
    lidar_available: torch.Tensor
    intrinsics: torch.Tensor
    camera_to_vehicle: torch.Tensor
    camera_to_world: torch.Tensor
    satellite_meters_per_pixel: torch.Tensor
    satellite_geometry_valid: torch.Tensor
    satellite_center_world_xy: torch.Tensor
    satellite_forward_world_xy: torch.Tensor
    satellite_right_world_xy: torch.Tensor
    satellite_is_ego_oriented: torch.Tensor
    explicit_negative_mask: torch.Tensor
    negative_available: torch.Tensor

    def __len__(self) -> int:
        return len(self.sample_ids)


def _load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_depth(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32)
    elif path.suffix == ".pt":
        array = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    else:
        array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.max() > 255:
            array = array / 256.0
    return torch.from_numpy(array.astype(np.float32))


def _load_mask(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return torch.from_numpy(array > 0)


class Stage1Dataset(Dataset):
    """Manifest-driven Stage-1 dataset independent from stage1_anchor code."""

    def __init__(self, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        payload = json.loads(manifest_path.read_text())
        self.records: list[Stage1SampleRecord] = []
        for item in payload:
            camera_to_vehicle = item.get("camera_to_vehicle") or item.get("camera_to_pose")
            vehicle_to_world = item.get("vehicle_to_world") or item.get("pose_to_world")
            if camera_to_vehicle is None or vehicle_to_world is None:
                raise KeyError(
                    "Manifest entry is missing camera/extrinsics. "
                    "Expected camera_to_vehicle/camera_to_pose and vehicle_to_world/pose_to_world."
                )
            self.records.append(
                Stage1SampleRecord(
                    front_rgb_path=self._resolve_path(item, "front_rgb_path", "front_image"),
                    satellite_rgb_path=self._resolve_path(item, "satellite_rgb_path", "satellite_image"),
                    depth_path=self._resolve_depth_path(item),
                    depth_valid_mask_path=self._resolve_optional_path(item, "depth_valid_mask_path", "unidepth_valid_mask_path"),
                    depth_confidence_path=self._resolve_optional_path(item, "depth_confidence_path", "unidepth_confidence_path"),
                    lidar_depth_path=self._resolve_optional_path(item, "lidar_depth_path", "depth"),
                    intrinsics=item["intrinsics"],
                    camera_to_vehicle=camera_to_vehicle,
                    vehicle_to_world=vehicle_to_world,
                    satellite_meters_per_pixel=float(item.get("satellite_meters_per_pixel", 0.2)),
                    satellite_geometry_valid=(
                        "satellite_meters_per_pixel" in item
                        and "satellite_center_world_xy" in item
                        and np.isfinite(float(item.get("satellite_meters_per_pixel", 0.0)))
                        and float(item.get("satellite_meters_per_pixel", 0.0)) > 0.0
                        and len(item.get("satellite_center_world_xy", ())) == 2
                        and np.isfinite(np.asarray(item.get("satellite_center_world_xy"), dtype=np.float32)).all()
                        and (
                            (item.get("satellite_forward_world_xy") is None and item.get("satellite_right_world_xy") is None)
                            or (item.get("satellite_forward_world_xy") is not None and item.get("satellite_right_world_xy") is not None)
                        )
                    ),
                    satellite_center_world_xy=tuple(item.get("satellite_center_world_xy", (0.0, 0.0))),
                    satellite_forward_world_xy=(
                        tuple(item["satellite_forward_world_xy"])
                        if item.get("satellite_forward_world_xy") is not None
                        else None
                    ),
                    satellite_right_world_xy=(
                        tuple(item["satellite_right_world_xy"])
                        if item.get("satellite_right_world_xy") is not None
                        else None
                    ),
                    negative_mask_path=self._resolve_optional_path(item, "negative_mask_path"),
                    sample_id=item.get("sample_id"),
                )
            )

    @staticmethod
    def _resolve_path(item: dict[str, Any], *keys: str) -> Path:
        for key in keys:
            value = item.get(key)
            if value:
                return Path(value)
        raise KeyError(f"Manifest entry is missing one of the required path keys: {keys}")

    @staticmethod
    def _resolve_optional_path(item: dict[str, Any], *keys: str) -> Path | None:
        for key in keys:
            value = item.get(key)
            if value:
                return Path(value)
        return None

    @staticmethod
    def _resolve_depth_path(item: dict[str, Any]) -> Path | None:
        value = (
            item.get("unidepth_depth_path")
            or item.get("front_depth_path")
            or item.get("depth_path")
            or item.get("da3_depth_path")
        )
        return Path(value) if value else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Stage1Sample:
        record = self.records[index]
        camera_to_vehicle = torch.tensor(record.camera_to_vehicle, dtype=torch.float32)
        vehicle_to_world = torch.tensor(record.vehicle_to_world, dtype=torch.float32)
        camera_to_world = vehicle_to_world @ camera_to_vehicle
        if record.satellite_forward_world_xy is None:
            forward_world_xy = camera_to_world[:2, 2]
        else:
            forward_world_xy = torch.tensor(record.satellite_forward_world_xy, dtype=torch.float32)
        if record.satellite_right_world_xy is None:
            right_world_xy = camera_to_world[:2, 0]
        else:
            right_world_xy = torch.tensor(record.satellite_right_world_xy, dtype=torch.float32)
        forward_world_xy = forward_world_xy / forward_world_xy.norm().clamp_min(1e-8)
        right_world_xy = right_world_xy / right_world_xy.norm().clamp_min(1e-8)
        sample_id = record.sample_id or record.front_rgb_path.stem
        front_rgb = _load_rgb(record.front_rgb_path)
        satellite_rgb = _load_rgb(record.satellite_rgb_path)
        da3_depth = (
            _load_depth(record.depth_path)
            if record.depth_path is not None and record.depth_path.exists()
            else None
        )
        if da3_depth is not None:
            depth_valid_mask = (
                _load_mask(record.depth_valid_mask_path)
                if record.depth_valid_mask_path is not None and record.depth_valid_mask_path.exists()
                else torch.isfinite(da3_depth) & (da3_depth > 0)
            )
            depth_confidence = (
                _load_depth(record.depth_confidence_path).clamp(0.0, 1.0)
                if record.depth_confidence_path is not None and record.depth_confidence_path.exists()
                else depth_valid_mask.float()
            )
            if depth_valid_mask.shape != da3_depth.shape:
                depth_valid_mask = F.interpolate(
                    depth_valid_mask[None, None].float(), da3_depth.shape, mode="nearest"
                )[0, 0] > 0.5
            if depth_confidence.shape != da3_depth.shape:
                depth_confidence = F.interpolate(
                    depth_confidence[None, None], da3_depth.shape, mode="bilinear", align_corners=False
                )[0, 0]
            depth_valid_mask &= torch.isfinite(da3_depth) & (da3_depth > 0)
            depth_confidence = depth_confidence.nan_to_num(0.0).clamp(0.0, 1.0) * depth_valid_mask
        else:
            depth_valid_mask = torch.zeros_like(front_rgb[0], dtype=torch.bool)
            depth_confidence = torch.zeros_like(front_rgb[0])
        return Stage1Sample(
            sample_id=sample_id,
            front_rgb=front_rgb,
            satellite_rgb=satellite_rgb,
            da3_depth=da3_depth,
            da3_depth_valid_mask=depth_valid_mask,
            da3_depth_confidence=depth_confidence,
            lidar_depth=_load_depth(record.lidar_depth_path) if record.lidar_depth_path else None,
            intrinsics=torch.tensor(record.intrinsics, dtype=torch.float32),
            camera_to_vehicle=camera_to_vehicle,
            vehicle_to_world=vehicle_to_world,
            camera_to_world=camera_to_world,
            satellite_meters_per_pixel=record.satellite_meters_per_pixel,
            satellite_geometry_valid=record.satellite_geometry_valid,
            satellite_center_world_xy=record.satellite_center_world_xy,
            satellite_forward_world_xy=tuple(float(value) for value in forward_world_xy.tolist()),
            satellite_right_world_xy=tuple(float(value) for value in right_world_xy.tolist()),
            satellite_is_ego_oriented=(
                record.satellite_forward_world_xy is not None
                and record.satellite_right_world_xy is not None
            ),
            explicit_negative_mask=_load_mask(record.negative_mask_path),
        )


class SafeStage1Dataset(Dataset):
    def __init__(self, dataset: Stage1Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Stage1Sample | SampleLoadError:
        try:
            return self.dataset[index]
        except Exception as exc:
            record = self.dataset.records[index]
            sample_id = record.sample_id or record.front_rgb_path.stem
            return SampleLoadError(
                index=index,
                sample_id=sample_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                front_rgb_path=str(record.front_rgb_path),
                satellite_rgb_path=str(record.satellite_rgb_path),
                depth_path=str(record.depth_path),
                lidar_depth_path=str(record.lidar_depth_path) if record.lidar_depth_path is not None else None,
            )


def projection_collate(samples: list[Stage1Sample | SampleLoadError]) -> tuple[Stage1ProjectionBatch | None, list[SampleLoadError]]:
    valid_samples = [sample for sample in samples if isinstance(sample, Stage1Sample)]
    load_errors = [sample for sample in samples if isinstance(sample, SampleLoadError)]
    if not valid_samples:
        return None, load_errors
    sample_ids = [sample.sample_id for sample in valid_samples]
    front_rgb = torch.stack([sample.front_rgb for sample in valid_samples], dim=0)
    satellite_rgb = torch.stack([sample.satellite_rgb for sample in valid_samples], dim=0)
    zero_depth = torch.zeros_like(valid_samples[0].front_rgb[0])
    da3_depth = torch.stack(
        [
            F.interpolate(sample.da3_depth[None, None], sample.front_rgb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            if sample.da3_depth is not None and sample.da3_depth.shape != sample.front_rgb.shape[-2:]
            else sample.da3_depth if sample.da3_depth is not None else zero_depth
            for sample in valid_samples
        ],
        dim=0,
    )
    da3_depth_valid_mask = torch.stack(
        [
            F.interpolate(sample.da3_depth_valid_mask[None, None].float(), sample.front_rgb.shape[-2:], mode="nearest")[0, 0] > 0.5
            if sample.da3_depth_valid_mask.shape != sample.front_rgb.shape[-2:]
            else sample.da3_depth_valid_mask
            for sample in valid_samples
        ],
        dim=0,
    )
    da3_depth_confidence = torch.stack(
        [
            F.interpolate(sample.da3_depth_confidence[None, None], sample.front_rgb.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            if sample.da3_depth_confidence.shape != sample.front_rgb.shape[-2:]
            else sample.da3_depth_confidence
            for sample in valid_samples
        ],
        dim=0,
    )
    lidar_available = torch.tensor([sample.lidar_depth is not None for sample in valid_samples], dtype=torch.bool)
    lidar_depth = torch.stack(
        [
            sample.lidar_depth if sample.lidar_depth is not None else zero_depth
            for sample in valid_samples
        ],
        dim=0,
    )
    negative_available = torch.tensor([sample.explicit_negative_mask is not None for sample in valid_samples], dtype=torch.bool)
    explicit_negative_mask = torch.stack(
        [
            sample.explicit_negative_mask
            if sample.explicit_negative_mask is not None
            else torch.zeros_like(zero_depth, dtype=torch.bool)
            for sample in valid_samples
        ],
        dim=0,
    )
    return Stage1ProjectionBatch(
        sample_ids=sample_ids,
        front_rgb=front_rgb,
        satellite_rgb=satellite_rgb,
        da3_depth=da3_depth,
        da3_depth_valid_mask=da3_depth_valid_mask,
        da3_depth_confidence=da3_depth_confidence,
        lidar_depth=lidar_depth,
        lidar_available=lidar_available,
        intrinsics=torch.stack([sample.intrinsics for sample in valid_samples], dim=0),
        camera_to_vehicle=torch.stack([sample.camera_to_vehicle for sample in valid_samples], dim=0),
        camera_to_world=torch.stack([sample.camera_to_world for sample in valid_samples], dim=0),
        satellite_meters_per_pixel=torch.tensor([sample.satellite_meters_per_pixel for sample in valid_samples], dtype=torch.float32),
        satellite_geometry_valid=torch.tensor([sample.satellite_geometry_valid for sample in valid_samples], dtype=torch.bool),
        satellite_center_world_xy=torch.tensor([sample.satellite_center_world_xy for sample in valid_samples], dtype=torch.float32),
        satellite_forward_world_xy=torch.tensor([sample.satellite_forward_world_xy for sample in valid_samples], dtype=torch.float32),
        satellite_right_world_xy=torch.tensor([sample.satellite_right_world_xy for sample in valid_samples], dtype=torch.float32),
        satellite_is_ego_oriented=torch.tensor(
            [sample.satellite_is_ego_oriented for sample in valid_samples], dtype=torch.bool
        ),
        explicit_negative_mask=explicit_negative_mask,
        negative_available=negative_available,
    ), load_errors


def build_dataloader(
    dataset: Stage1Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: Sampler | None = None,
) -> DataLoader:
    return DataLoader(
        SafeStage1Dataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=projection_collate,
        sampler=sampler,
    )


def move_batch_to_device(batch: Stage1ProjectionBatch, device: str | torch.device) -> Stage1ProjectionBatch:
    return Stage1ProjectionBatch(
        sample_ids=batch.sample_ids,
        front_rgb=batch.front_rgb.to(device),
        satellite_rgb=batch.satellite_rgb.to(device),
        da3_depth=batch.da3_depth.to(device),
        da3_depth_valid_mask=batch.da3_depth_valid_mask.to(device),
        da3_depth_confidence=batch.da3_depth_confidence.to(device),
        lidar_depth=batch.lidar_depth.to(device),
        lidar_available=batch.lidar_available.to(device),
        intrinsics=batch.intrinsics.to(device),
        camera_to_vehicle=batch.camera_to_vehicle.to(device),
        camera_to_world=batch.camera_to_world.to(device),
        satellite_meters_per_pixel=batch.satellite_meters_per_pixel.to(device),
        satellite_geometry_valid=batch.satellite_geometry_valid.to(device),
        satellite_center_world_xy=batch.satellite_center_world_xy.to(device),
        satellite_forward_world_xy=batch.satellite_forward_world_xy.to(device),
        satellite_right_world_xy=batch.satellite_right_world_xy.to(device),
        satellite_is_ego_oriented=batch.satellite_is_ego_oriented.to(device),
        explicit_negative_mask=batch.explicit_negative_mask.to(device),
        negative_available=batch.negative_available.to(device),
    )
