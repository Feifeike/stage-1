from __future__ import annotations

import hashlib
import os
from pathlib import Path


def checkpoint_path_with_progress(template: Path, epoch: int, step: int) -> Path:
    """Return a checkpoint path whose filename records its training progress."""

    if template.suffix:
        name = f"{template.stem}_epoch_{int(epoch):03d}_step_{int(step):06d}{template.suffix}"
    else:
        name = f"{template.name}_epoch_{int(epoch):03d}_step_{int(step):06d}"
    return template.with_name(name)


def update_checkpoint_alias(alias: Path, target: Path, *, keep_previous_versions: bool) -> None:
    """Atomically point a legacy stable name at a progress-named checkpoint.

    Latest aliases retain the same single-snapshot behavior as the old
    overwritten ``*_latest.pth`` files. Best-checkpoint aliases can preserve
    every progress-named best snapshot.
    """

    alias.parent.mkdir(parents=True, exist_ok=True)
    previous_target = None
    if alias.is_symlink():
        previous_target = alias.resolve(strict=False)
    elif keep_previous_versions and alias.is_file():
        # Preserve a pre-versioning best checkpoint before converting its
        # stable filename into an alias. Native project checkpoints carry the
        # progress metadata needed to give the preserved file an honest name.
        import torch

        payload = torch.load(alias, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "epoch" not in payload or "step" not in payload:
            raise ValueError(
                f"Cannot version the existing best checkpoint {alias}: it has no epoch/step metadata."
            )
        epoch = int(payload["epoch"])
        step = int(payload["step"])
        legacy_name = f"{alias.stem}_legacy_epoch_{epoch:03d}_step_{step:06d}{alias.suffix}"
        archived = alias.with_name(legacy_name)
        version = 2
        while archived.exists():
            legacy_name = (
                f"{alias.stem}_legacy_{version}_epoch_{epoch:03d}_step_{step:06d}{alias.suffix}"
            )
            archived = alias.with_name(legacy_name)
            version += 1
        os.replace(alias, archived)

    temporary = alias.with_name(f".{alias.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.name)
    os.replace(temporary, alias)

    if (
        keep_previous_versions
        or previous_target is None
        or previous_target == target.resolve(strict=False)
    ):
        return
    expected_prefix = f"{alias.stem}_epoch_"
    if (
        previous_target.parent == alias.parent.resolve()
        and previous_target.name.startswith(expected_prefix)
        and previous_target.suffix == alias.suffix
        and previous_target.exists()
    ):
        previous_target.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
