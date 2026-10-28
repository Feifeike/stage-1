from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def cross(image: np.ndarray, x: float, y: float, color: tuple[int, int, int], size: int = 9) -> None:
    u, v = int(round(x)), int(round(y))
    cv2.drawMarker(image, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_depth_m", type=float, default=80.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    record = next(item for item in manifest if str(item["sample_id"]) == str(args.sample_id))
    matches = json.loads(args.matches.read_text())
    match = next(item for item in matches if str(item["sampleId"]) == str(args.sample_id))

    front_path = Path(record["front_rgb_path"])
    satellite_path = Path(record["satellite_rgb_path"])
    depth_path = Path(record["lidar_depth_path"])
    front = cv2.imread(str(front_path), cv2.IMREAD_COLOR)
    satellite = cv2.imread(str(satellite_path), cv2.IMREAD_COLOR)
    depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
    depth_m = depth_raw / 256.0
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= float(args.max_depth_m))

    intrinsics = np.asarray(record["intrinsics"], dtype=np.float64)
    camera_to_world = np.asarray(record["vehicle_to_world"], dtype=np.float64)
    mpp = float(record.get("satellite_meters_per_pixel", 0.2))
    # The collected satellite image is ego-oriented: vehicle position is at
    # the tile center and vehicle forward points toward decreasing v.
    pose_path = Path(record["metadata"]["pose_source"])
    pose_timestamp = int(record["metadata"]["pose_timestamp"])
    pose_row = next(
        line.split()
        for line in pose_path.read_text().splitlines()
        if line.split() and int(line.split()[0]) == pose_timestamp
    )
    px, py = float(pose_row[1]), float(pose_row[2])
    yaw = float(pose_row[6])
    center_world = np.asarray([px, py], dtype=np.float64)
    forward_world = np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    right_world = np.asarray([np.sin(yaw), -np.cos(yaw)], dtype=np.float64)
    front_h, front_w = depth_m.shape
    sat_h, sat_w = satellite.shape[:2]

    vv, uu = np.nonzero(valid)
    z = depth_m[vv, uu].astype(np.float64)
    pixels = np.stack([uu.astype(np.float64), vv.astype(np.float64), np.ones_like(z)], axis=1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    points_camera = rays * z[:, None]
    points_world = points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    delta_world = points_world[:, :2] - center_world[None, :]
    sat_u = delta_world @ right_world / mpp + (sat_w - 1) * 0.5
    sat_v = (sat_h - 1) * 0.5 - delta_world @ forward_world / mpp
    sat_in = (sat_u >= 0) & (sat_u < sat_w) & (sat_v >= 0) & (sat_v < sat_h)

    # Depth-colored LiDAR points on the front image.
    color_norm = np.clip(depth_m / float(args.max_depth_m) * 255.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(color_norm, cv2.COLORMAP_TURBO)
    front_overlay = front.copy()
    front_points = np.zeros_like(front)
    front_points[vv, uu] = color_map[vv, uu]
    front_overlay = cv2.addWeighted(front_overlay, 0.72, front_points, 0.95, 0)

    # Depth-colored LiDAR points on the satellite tile.
    sat_overlay = satellite.copy()
    sat_points = np.zeros_like(satellite)
    colors = color_map[vv, uu]
    sat_u_i = np.rint(sat_u[sat_in]).astype(np.int32)
    sat_v_i = np.rint(sat_v[sat_in]).astype(np.int32)
    sat_points[sat_v_i, sat_u_i] = colors[sat_in]
    sat_overlay = cv2.addWeighted(sat_overlay, 0.72, sat_points, 1.0, 0)

    # Mark the camera/world center in the satellite tile.
    cross(sat_overlay, (sat_w - 1) * 0.5, (sat_h - 1) * 0.5, (255, 255, 255), 20)
    cv2.putText(sat_overlay, "vehicle center", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mark the same selected query anchors in the front and their GT/coarse/refine locations in the tile.
    for index, anchor in enumerate(match.get("anchors", []), start=1):
        x, y = float(anchor["x"]), float(anchor["y"])
        cv2.circle(front_overlay, (int(round(x)), int(round(y))), 9, (0, 255, 255), 2)
        cv2.putText(front_overlay, str(index), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        # Recompute this anchor's GT from the local LiDAR depth using the same
        # 4 px search radius as the evaluator, but with ego-oriented tile axes.
        u0, v0 = int(round(x)), int(round(y))
        best = None
        for radius in range(0, 5):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    uu0 = min(max(u0 + dx, 0), front_w - 1)
                    vv0 = min(max(v0 + dy, 0), front_h - 1)
                    z0 = float(depth_m[vv0, uu0])
                    if z0 > 0 and z0 <= float(args.max_depth_m):
                        best = (uu0, vv0, z0)
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is not None:
            uu0, vv0, z0 = best
            ray0 = np.linalg.inv(intrinsics) @ np.asarray([uu0, vv0, 1.0], dtype=np.float64)
            point_world0 = camera_to_world[:3, :3] @ (ray0 * z0) + camera_to_world[:3, 3]
            delta0 = point_world0[:2] - center_world
            gt_u = float(delta0 @ right_world / mpp + (sat_w - 1) * 0.5)
            gt_v = float((sat_h - 1) * 0.5 - delta0 @ forward_world / mpp)
            cross(sat_overlay, gt_u, gt_v, (0, 210, 80), 15)
        if anchor.get("gt_x") is not None:
            cross(sat_overlay, anchor["coarse_x"], anchor["coarse_y"], (0, 180, 255), 13)
            cross(sat_overlay, anchor["pred_x"], anchor["pred_y"], (0, 0, 255), 15)

    cv2.putText(front_overlay, "LiDAR GT projection on front view; yellow circles=query anchors", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(sat_overlay, "LiDAR GT=green cloud/green crosses; coarse=orange; refine=red", (10, sat_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    front_out = args.output_dir / f"{args.sample_id}_front_gt.png"
    sat_out = args.output_dir / f"{args.sample_id}_satellite_gt.png"
    cv2.imwrite(str(front_out), front_overlay)
    cv2.imwrite(str(sat_out), sat_overlay)

    # Put both views into one file for quick inspection.
    sat_resized = cv2.resize(sat_overlay, (front_w, front_h), interpolation=cv2.INTER_NEAREST)
    combined = np.concatenate([front_overlay, sat_resized], axis=1)
    combined_out = args.output_dir / f"{args.sample_id}_gt_projection_combined.jpg"
    cv2.imwrite(str(combined_out), combined, [cv2.IMWRITE_JPEG_QUALITY, 94])

    stats = {
        "sample_id": str(args.sample_id),
        "front_image": str(front_path),
        "satellite_image": str(satellite_path),
        "depth_image": str(depth_path),
        "depth_valid_points": int(valid.sum()),
        "depth_min_m": float(z.min()) if z.size else None,
        "depth_max_m": float(z.max()) if z.size else None,
        "satellite_in_bounds_points": int(sat_in.sum()),
        "satellite_in_bounds_fraction": float(sat_in.mean()) if sat_in.size else 0.0,
        "selected_anchors": len(match.get("anchors", [])),
        "selected_anchors_with_gt": sum(anchor.get("gt_x") is not None for anchor in match.get("anchors", [])),
        "satellite_orientation": "ego-oriented: vehicle center at tile center, vehicle forward up",
        "outputs": [str(front_out), str(sat_out), str(combined_out)],
    }
    (args.output_dir / f"{args.sample_id}_projection_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
