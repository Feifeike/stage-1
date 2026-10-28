from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Stage1Targets:
    query_uv: torch.Tensor
    gt_sat_uv: torch.Tensor
    gt_world_xy: torch.Tensor
    lidar_z: torch.Tensor
    valid_gt_mask: torch.Tensor
    matchability_targets: torch.Tensor
    matchability_supervised_mask: torch.Tensor


def pixels_to_camera_rays(pixels_uv: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Convert pixels to OpenCV camera rays with unit z.

    Camera convention: x right, y down, z forward. A point with camera z-depth
    `z` is `z * [(u-cx)/fx, (v-cy)/fy, 1]`.
    """

    if pixels_uv.shape[-1] != 2:
        raise ValueError("pixels_uv must end with 2 values [u, v].")
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    u = pixels_uv[..., 0]
    v = pixels_uv[..., 1]
    if intrinsics.ndim == 3:
        view_shape = intrinsics.shape[:-2] + (1,) * (u.ndim - intrinsics.ndim + 2)
        fx = fx.reshape(view_shape)
        fy = fy.reshape(view_shape)
        cx = cx.reshape(view_shape)
        cy = cy.reshape(view_shape)
    x = (u - cx) / fx
    y = (v - cy) / fy
    return torch.stack((x, y, torch.ones_like(x)), dim=-1)


def make_query_grid(
    batch_size: int,
    feat_hw: tuple[int, int],
    image_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return feature-grid query centers in front-image pixel coordinates.

    Shape: `[B, Hf*Wf, 2]`, values are `(u, v)` in original front-image pixels.
    """

    feat_h, feat_w = feat_hw
    image_h, image_w = image_hw
    ys = torch.arange(feat_h, device=device, dtype=dtype)
    xs = torch.arange(feat_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if feat_w > 1:
        u = grid_x * ((image_w - 1) / (feat_w - 1))
    else:
        u = torch.full_like(grid_x, (image_w - 1) * 0.5)
    if feat_h > 1:
        v = grid_y * ((image_h - 1) / (feat_h - 1))
    else:
        v = torch.full_like(grid_y, (image_h - 1) * 0.5)
    uv = torch.stack((u, v), dim=-1).reshape(1, feat_h * feat_w, 2)
    return uv.expand(batch_size, -1, -1).contiguous()


def make_satellite_grid(
    feat_hw: tuple[int, int],
    image_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return satellite feature-grid centers in satellite-image pixel units.

    Shape: `[Hs*Ws, 2]`, values are `(u, v)` in original satellite-image pixels.
    """

    feat_h, feat_w = feat_hw
    image_h, image_w = image_hw
    ys = torch.arange(feat_h, device=device, dtype=dtype)
    xs = torch.arange(feat_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if feat_w > 1:
        u = grid_x * ((image_w - 1) / (feat_w - 1))
    else:
        u = torch.full_like(grid_x, (image_w - 1) * 0.5)
    if feat_h > 1:
        v = grid_y * ((image_h - 1) / (feat_h - 1))
    else:
        v = torch.full_like(grid_y, (image_h - 1) * 0.5)
    return torch.stack((u, v), dim=-1).reshape(feat_h * feat_w, 2).contiguous()


def sample_sparse_lidar_at_queries(
    lidar_depth: torch.Tensor,
    query_uv: torch.Tensor,
    radius_px: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample z-depth for query pixels from a sparse LiDAR depth image.

    Shape:
    - `lidar_depth`: `[B, H, W]`, camera z-depth in meters, zero means unknown.
    - `query_uv`: `[B, N, 2]`, front-image pixel coordinates.

    If `radius_px > 0`, the nearest valid LiDAR point in a square local window is
    used. Unknown queries remain invalid; they are not negatives.
    """

    batch_size, image_h, image_w = lidar_depth.shape
    u = query_uv[..., 0].round().long().clamp(0, image_w - 1)
    v = query_uv[..., 1].round().long().clamp(0, image_h - 1)
    if radius_px <= 0:
        batch_ids = torch.arange(batch_size, device=lidar_depth.device)[:, None]
        z = lidar_depth[batch_ids, v, u]
        return z, torch.isfinite(z) & (z > 0)

    z_out = torch.zeros(query_uv.shape[:2], dtype=lidar_depth.dtype, device=lidar_depth.device)
    valid_out = torch.zeros(query_uv.shape[:2], dtype=torch.bool, device=lidar_depth.device)
    offsets = torch.arange(-radius_px, radius_px + 1, device=lidar_depth.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_sq = (dx.square() + dy.square()).reshape(-1)
    order = torch.argsort(offset_sq)
    dx = dx.reshape(-1)[order]
    dy = dy.reshape(-1)[order]

    for ox, oy in zip(dx.tolist(), dy.tolist()):
        uu = (u + ox).clamp(0, image_w - 1)
        vv = (v + oy).clamp(0, image_h - 1)
        batch_ids = torch.arange(batch_size, device=lidar_depth.device)[:, None]
        z = lidar_depth[batch_ids, vv, uu]
        take = (~valid_out) & torch.isfinite(z) & (z > 0)
        z_out = torch.where(take, z, z_out)
        valid_out = valid_out | take
        if bool(valid_out.all().item()):
            break
    return z_out, valid_out


def build_stage1_targets(
    *,
    lidar_depth: torch.Tensor,
    query_uv: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    satellite_meters_per_pixel: torch.Tensor,
    satellite_center_world_xy: torch.Tensor,
    satellite_image_hw: tuple[int, int],
    explicit_negative_mask: torch.Tensor | None = None,
    lidar_query_radius_px: int = 0,
) -> Stage1Targets:
    """Build geometry supervision for semi-dense front queries.

    Output coordinates are in original satellite-image pixel units. All masks keep
    unknown and explicit negative queries distinct.
    """

    batch_size, num_queries, _ = query_uv.shape
    sat_h, sat_w = satellite_image_hw
    z_depth, has_lidar = sample_sparse_lidar_at_queries(
        lidar_depth=lidar_depth,
        query_uv=query_uv,
        radius_px=lidar_query_radius_px,
    )
    rays = pixels_to_camera_rays(query_uv, intrinsics)
    points_camera = rays * z_depth[..., None]
    rotation = camera_to_world[:, :3, :3]
    origin = camera_to_world[:, :3, 3]
    points_world = torch.matmul(points_camera, rotation.transpose(1, 2)) + origin[:, None, :]
    world_xy = points_world[..., :2]

    mpp = satellite_meters_per_pixel.to(query_uv.dtype).view(batch_size, 1)
    center_xy = satellite_center_world_xy.to(query_uv.dtype)
    center_u = (sat_w - 1) * 0.5
    center_v = (sat_h - 1) * 0.5
    sat_u = (world_xy[..., 0] - center_xy[:, None, 0]) / mpp + center_u
    sat_v = center_v - (world_xy[..., 1] - center_xy[:, None, 1]) / mpp
    gt_sat_uv = torch.stack((sat_u, sat_v), dim=-1)
    finite_projection = torch.isfinite(gt_sat_uv).all(dim=-1) & torch.isfinite(world_xy).all(dim=-1)
    in_satellite = (
        (gt_sat_uv[..., 0] >= 0)
        & (gt_sat_uv[..., 0] <= sat_w - 1)
        & (gt_sat_uv[..., 1] >= 0)
        & (gt_sat_uv[..., 1] <= sat_h - 1)
    )
    valid_gt = has_lidar & finite_projection & in_satellite
    projected_outside_crop = has_lidar & finite_projection & ~in_satellite

    matchability_targets = torch.zeros((batch_size, num_queries), dtype=query_uv.dtype, device=query_uv.device)
    supervised = valid_gt | projected_outside_crop
    matchability_targets = torch.where(valid_gt, torch.ones_like(matchability_targets), matchability_targets)
    if explicit_negative_mask is not None:
        neg = F.grid_sample(
            explicit_negative_mask[:, None].float(),
            _uv_to_grid(query_uv, explicit_negative_mask.shape[-2:]).unsqueeze(2),
            mode="nearest",
            align_corners=True,
        ).squeeze(1).squeeze(-1) > 0.5
        neg = neg & ~valid_gt
        supervised = supervised | neg
        matchability_targets = torch.where(neg, torch.zeros_like(matchability_targets), matchability_targets)

    return Stage1Targets(
        query_uv=query_uv,
        gt_sat_uv=gt_sat_uv,
        gt_world_xy=world_xy,
        lidar_z=z_depth,
        valid_gt_mask=valid_gt,
        matchability_targets=matchability_targets,
        matchability_supervised_mask=supervised,
    )


def _uv_to_grid(uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = image_hw
    x = uv[..., 0] / max(image_w - 1, 1) * 2.0 - 1.0
    y = uv[..., 1] / max(image_h - 1, 1) * 2.0 - 1.0
    return torch.stack((x, y), dim=-1)
