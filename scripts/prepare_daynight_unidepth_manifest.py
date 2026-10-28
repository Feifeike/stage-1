from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unidepth_dir", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_records = []
    missing = []
    for item in records:
        sample_id = str(item["sample_id"])
        depth_path = args.unidepth_dir / f"{sample_id}.npy"
        if not depth_path.exists():
            missing.append(sample_id)
            continue
        new_item = copy.deepcopy(item)
        # Stage1Dataset prioritizes unidepth_depth_path for da3_depth.
        new_item["unidepth_depth_path"] = str(depth_path)
        new_item["depth_path"] = str(depth_path)
        metadata = dict(new_item.get("metadata", {}))
        metadata["unidepth_depth_source"] = str(depth_path)
        metadata["lidar_depth_source"] = str(item["lidar_depth_path"])
        new_item["metadata"] = metadata
        output_records.append(new_item)

    if missing:
        raise FileNotFoundError(
            f"Missing UniDepth predictions for {len(missing)} samples; first={missing[:5]}"
        )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(output_records),
                "unidepth_dir": str(args.unidepth_dir),
                "output_manifest": str(args.output_manifest),
                "lidar_depth_preserved": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
