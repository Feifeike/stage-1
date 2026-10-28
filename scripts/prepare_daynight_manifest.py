from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def nearest_path(paths: list[Path], timestamp: int) -> tuple[Path, int]:
    path = min(paths, key=lambda item: abs(int(item.stem) - timestamp))
    return path, int(path.stem) - timestamp


def pose_matrix(row: list[str]) -> np.ndarray:
    _, x, y, z, roll, pitch, yaw = row[:7]
    roll, pitch, yaw = float(roll), float(pitch), float(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = [float(x), float(y), float(z)]
    return transform


def read_camera_calibration(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # These values are the calibrated camera matrix and LiDAR/IMU extrinsics in
    # para/*.ini.  The projection matrix is used to recover camera_from_lidar.
    k = np.array(
        [[1957.43694246195, 0.0, 983.756456853235],
         [0.0, 1956.52145070368, 460.726049215283],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projection = np.array(
        [[1007.0, -1943.23, 95.6702, -51418.9],
         [577.403, -28.8039, -1925.1, -126289.0],
         [0.998111, 0.0139286, 0.059833, 30.5194]],
        dtype=np.float64,
    )
    # P was used with LiDAR points converted to millimetres in the original
    # depth projection script, so its translation is in millimetres. Convert
    # that translation to metres before composing it with the SLAM pose.
    camera_from_lidar = np.linalg.solve(k, projection)
    camera_from_lidar[:3, 3] /= 1000.0
    r_imu_from_lidar = np.array(
        [[0.990827, -0.129304, -0.0392663],
         [0.128605, 0.991497, -0.0198371],
         [0.0414974, 0.0146053, 0.999032]],
        dtype=np.float64,
    )
    t_imu_from_lidar = np.array([2.00178, -0.00286412, 1.7994], dtype=np.float64)
    return k, camera_from_lidar[:3, :3], camera_from_lidar[:3, 3], np.block(
        [[r_imu_from_lidar, t_imu_from_lidar[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=400)
    args = parser.parse_args()

    root = args.root
    depth_paths = sorted((root / "depth_projected").glob("*.png"), key=lambda p: int(p.stem))
    front_paths = sorted((root / "aligned_Image").glob("*.jpg"), key=lambda p: int(p.stem))
    satellite_paths = sorted((root / "Satellite").glob("*.jpg"), key=lambda p: int(p.stem))
    pose_path = root / "output" / "pose_slam_euler_aligned.txt"
    pose_rows = {
        int(row[0]): row
        for row in (line.split() for line in pose_path.read_text().splitlines())
        if len(row) >= 7
    }
    if args.num_samples <= 0 or args.num_samples > len(depth_paths):
        raise ValueError(f"num_samples must be in [1, {len(depth_paths)}]")
    if args.num_samples == 1:
        selected_depth = [depth_paths[0]]
    else:
        selected_depth = [
            depth_paths[round(index * (len(depth_paths) - 1) / (args.num_samples - 1))]
            for index in range(args.num_samples)
        ]

    k, rotation_camera_from_lidar, translation_camera_from_lidar, imu_from_lidar = read_camera_calibration(root)
    camera_from_lidar = np.eye(4, dtype=np.float64)
    camera_from_lidar[:3, :3] = rotation_camera_from_lidar
    camera_from_lidar[:3, 3] = translation_camera_from_lidar
    lidar_from_camera = np.linalg.inv(camera_from_lidar)
    camera_records = []
    pairing_deltas = []
    for depth_path in selected_depth:
        timestamp = int(depth_path.stem)
        front_path, front_delta = nearest_path(front_paths, timestamp)
        satellite_path, satellite_delta = nearest_path(satellite_paths, timestamp)
        if timestamp not in pose_rows:
            pose_timestamp = min(pose_rows, key=lambda value: abs(value - timestamp))
            pose_row = pose_rows[pose_timestamp]
        else:
            pose_timestamp = timestamp
            pose_row = pose_rows[timestamp]
        world_from_imu = pose_matrix(pose_row)
        world_from_camera = world_from_imu @ imu_from_lidar @ lidar_from_camera
        camera_records.append(
            {
                "sample_id": str(timestamp),
                "front_rgb_path": str(front_path),
                "satellite_rgb_path": str(satellite_path),
                "depth_path": str(depth_path),
                "lidar_depth_path": str(depth_path),
                "intrinsics": k.tolist(),
                "camera_to_vehicle": np.eye(4, dtype=np.float64).tolist(),
                "vehicle_to_world": world_from_camera.tolist(),
                "satellite_meters_per_pixel": 0.2,
                # The collected satellite crop is ego-oriented: the vehicle
                # pose is at the image center and vehicle forward points up.
                "satellite_center_world_xy": world_from_imu[:2, 3].tolist(),
                "satellite_forward_world_xy": world_from_imu[:2, 0].tolist(),
                "satellite_right_world_xy": (-world_from_imu[:2, 1]).tolist(),
                "metadata": {
                    "front_timestamp_delta_ms": front_delta,
                    "satellite_timestamp_delta_ms": satellite_delta,
                    "pose_timestamp": pose_timestamp,
                    "pose_source": str(pose_path),
                    "depth_encoding": "uint16_depth_meters_times_256",
                    "satellite_orientation": "ego_vehicle_forward_up",
                },
            }
        )
        pairing_deltas.append((front_delta, satellite_delta))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(camera_records, indent=2), encoding="utf-8")
    max_front_delta = max(abs(value[0]) for value in pairing_deltas)
    max_satellite_delta = max(abs(value[1]) for value in pairing_deltas)
    print(
        json.dumps(
            {
                "records": len(camera_records),
                "source_depth_frames": len(depth_paths),
                "max_front_pair_delta_ms": max_front_delta,
                "max_satellite_pair_delta_ms": max_satellite_delta,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
