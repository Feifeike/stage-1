from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from PIL import Image


def center_crop_box(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    if width < size or height < size:
        raise ValueError(f"Cannot crop {size}x{size} from {width}x{height}")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--crop_size", type=int, default=512)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_records = []
    processed = {}

    for item in records:
        source = Path(item["satellite_rgb_path"])
        output_path = args.output_dir / f"{source.stem}.png"
        source_key = str(source.resolve())
        if source_key not in processed:
            with Image.open(source) as image:
                image = image.convert("RGB")
                box = center_crop_box(image.width, image.height, args.crop_size)
                cropped = image.crop(box)
                cropped.save(output_path, format="PNG")
                processed[source_key] = {
                    "source_size": [image.width, image.height],
                    "crop_box_xyxy": list(box),
                }

        new_item = copy.deepcopy(item)
        new_item["satellite_rgb_path"] = str(output_path)
        metadata = dict(new_item.get("metadata", {}))
        metadata.update(
            {
                "satellite_crop": "center_crop",
                "satellite_crop_size_px": args.crop_size,
                "satellite_crop_source": str(source),
                "satellite_crop_box_xyxy": processed[source_key]["crop_box_xyxy"],
                "satellite_meters_per_pixel": float(item.get("satellite_meters_per_pixel", 0.2)),
            }
        )
        new_item["metadata"] = metadata
        output_records.append(new_item)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(output_records),
                "unique_satellite_images": len(processed),
                "crop_size": [args.crop_size, args.crop_size],
                "meters_per_pixel": float(records[0].get("satellite_meters_per_pixel", 0.2)) if records else None,
                "output_manifest": str(args.output_manifest),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
