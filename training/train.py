from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from stage1.utils.config import load_config
from stage1.utils.checkpointing import checkpoint_path_with_progress, update_checkpoint_alias
from stage1.data.dataset_adapter import (
    SafeStage1Dataset,
    Stage1Dataset,
    build_dataloader,
    move_batch_to_device,
    projection_collate,
)
from stage1.training.losses import compute_losses
from stage1.models.model import Stage1CorrespondenceModel
from stage1.scripts.visualize import (
    save_correspondence_visualization,
    save_interactive_correspondence_visualization,
    save_similarity_heatmap,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("models/stage1/configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--stage", choices=("coarse", "refine", "joint"), default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--overfit_batches", type=int, default=None)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    completed_epoch: bool,
    best_val_loss: float | None = None,
    best_val_metric: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "completed_epoch": bool(completed_epoch),
            "best_val_loss": best_val_loss,
            "best_val_metric": best_val_metric,
            "training_stage": str(config.get("training", {}).get("stage", "joint")),
            "config": config,
        },
        path,
    )


def _checkpoint_path_with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix:
        filename = f"{path.stem}{suffix}{path.suffix}"
    else:
        filename = f"{path.name}{suffix}"
    return path.with_name(filename)


def _save_versioned_checkpoint(
    path_template: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    *,
    keep_previous_versions: bool,
    **metadata,
) -> Path:
    checkpoint_path = checkpoint_path_with_progress(path_template, epoch, step)
    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scaler,
        epoch,
        step,
        config,
        **metadata,
    )
    update_checkpoint_alias(
        path_template,
        checkpoint_path,
        keep_previous_versions=keep_previous_versions,
    )
    return checkpoint_path


def _metric_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _tensorboard_scalar_tag(key: str) -> str:
    if key.startswith("certainty_"):
        return f"certainty/{key.removeprefix('certainty_')}"
    if key.startswith("matchability_"):
        return f"matchability/{key.removeprefix('matchability_')}"
    if key.startswith("coarse_"):
        return f"coarse/{key.removeprefix('coarse_')}"
    if key.startswith("loss_"):
        return f"loss/{key.removeprefix('loss_')}"
    if key.startswith("num_"):
        return f"num/{key.removeprefix('num_')}"
    if key == "refine_pixel_error":
        return "refine/error/pixel"
    if key == "median_refine_pixel_error":
        return "refine/error/median_pixel"
    if key == "mean_metric_error":
        return "refine/error/metric"
    if key == "median_metric_error":
        return "refine/error/median_metric"
    if key.endswith("_error"):
        return f"error/{key.removesuffix('_error')}"
    return key


def _metrics_path(output_dir: Path) -> Path:
    """Return a metrics path without mixing the legacy and current schemas."""

    path = output_dir / "metrics.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return path
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = handle.read().splitlines()
        last_record = json.loads(lines[-1]) if lines else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return path

    legacy_keys = {
        "matchability_confidence_mean",
        "coarse_confidence_mean",
        "refine_certainty_mean",
        "loss_coarse_confidence",
        "loss_refine_certainty",
    }
    if legacy_keys.intersection(last_record):
        return output_dir / "metrics_v2.jsonl"
    return path


def _configure_training_stage(model: Stage1CorrespondenceModel, stage: str) -> list[torch.nn.Parameter]:
    if stage not in {"coarse", "refine", "joint"}:
        raise ValueError("training.stage must be one of: coarse, refine, joint.")
    model.set_training_stage(stage)
    if stage == "coarse":
        model.matcher.requires_grad_(True)
        model.refiner.requires_grad_(False)
    elif stage == "refine":
        model.backbone.requires_grad_(False)
        model.matcher.requires_grad_(False)
        model.refiner.requires_grad_(True)
    else:
        model.matcher.requires_grad_(True)
        model.refiner.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def _scene_name(record) -> str:
    for part in reversed(record.front_rgb_path.parts):
        if part.endswith("_sync"):
            return part
    return record.front_rgb_path.parent.parent.parent.name


def _select_fixed_visualization_samples(
    dataset: Stage1Dataset,
    frames_per_scene: int,
) -> list[tuple[str, int]]:
    """Select deterministic, evenly spaced manifest indices for each scene."""

    if frames_per_scene <= 0:
        return []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[_scene_name(record)].append(index)

    selected: list[tuple[str, int]] = []
    for scene, indices in sorted(grouped.items()):
        if len(indices) <= frames_per_scene:
            chosen = indices
        elif frames_per_scene == 1:
            chosen = [indices[len(indices) // 2]]
        else:
            positions = [
                round(position * (len(indices) - 1) / (frames_per_scene - 1))
                for position in range(frames_per_scene)
            ]
            chosen = [indices[position] for position in positions]
        selected.extend((scene, index) for index in chosen)
    return selected


@torch.no_grad()
def _save_fixed_visualizations(
    model: Stage1CorrespondenceModel,
    dataset: Stage1Dataset,
    selected_samples: list[tuple[str, int]],
    config: dict,
    device: str,
    use_amp: bool,
    output_dir: Path,
    epoch: int,
) -> None:
    """Save the same fixed frames for every epoch under an epoch-specific folder."""

    if not selected_samples:
        return
    was_training = model.training
    model.eval()
    safe_dataset = SafeStage1Dataset(dataset)
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for scene, sample_index in selected_samples:
        selected = safe_dataset[sample_index]
        batch_payload, load_errors = projection_collate([selected])
        if batch_payload is None:
            error = load_errors[0] if load_errors else None
            detail = f": {error.error_type}: {error.error_message}" if error else ""
            print(
                f"skipping visualization scene={scene} index={sample_index}{detail}",
                flush=True,
            )
            continue
        batch = move_batch_to_device(batch_payload, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(batch)
        sample_id = batch.sample_ids[0]
        stem = f"{scene}_{sample_id}"
        save_correspondence_visualization(batch, output, epoch_dir / f"{stem}.png")
        save_similarity_heatmap(batch, output, epoch_dir / f"{stem}_heatmap.png")
        save_interactive_correspondence_visualization(
            batch,
            output,
            epoch_dir / f"{stem}_interactive.html",
        )
    if was_training:
        model.train()


@torch.no_grad()
def _run_validation(
    model: Stage1CorrespondenceModel,
    loader,
    config: dict,
    device: str,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(total=total, desc="validation", leave=False) if tqdm is not None else None
    for batch_idx, (batch_payload, _load_errors) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch_payload is None:
            if progress is not None:
                progress.update(1)
            continue
        batch = move_batch_to_device(batch_payload, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(batch)
            losses = compute_losses(output, config)
        for key, value in losses.metrics.items():
            totals[key] += _metric_to_float(value)
        count += 1
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    model.train()
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None and args.init_checkpoint is not None:
        raise ValueError("--checkpoint resumes full training state; use either --checkpoint or --init_checkpoint, not both.")
    config = load_config(args.config)
    if args.stage is not None:
        config.setdefault("training", {})["stage"] = args.stage
    output_dir = args.output_dir or Path(config["training"].get("output_dir", "outputs/stage1"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = _metrics_path(output_dir)
    legacy_metrics_schema = metrics_path.name != "metrics.jsonl"
    vis_dir = output_dir / "vis"
    val_vis_dir = output_dir / "val_vis"
    ckpt_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / ("tensorboard_v2" if legacy_metrics_schema else "tensorboard")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if SummaryWriter is not None else _NullSummaryWriter()
    if legacy_metrics_schema:
        print(
            f"legacy metrics schema detected; writing new records to {metrics_path.name} "
            f"and TensorBoard events to {tensorboard_dir.name}",
            flush=True,
        )

    dataset = Stage1Dataset(config["data"]["train_manifest"])
    batch_size = int(config["training"].get("batch_size", 1))
    shuffle = args.overfit_batches is None
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    if args.overfit_batches is not None:
        cached = []
        for idx, item in enumerate(loader):
            cached.append(item)
            if idx + 1 >= args.overfit_batches:
                break
        loader_iterable = cached
    else:
        loader_iterable = loader

    val_dataset = None
    val_loader = None
    val_manifest = config.get("data", {}).get("val_manifest")
    if val_manifest and not args.skip_validation:
        val_dataset = Stage1Dataset(val_manifest)
        val_loader = build_dataloader(
            val_dataset,
            batch_size=int(config.get("evaluation", {}).get("batch_size", batch_size)),
            shuffle=False,
            num_workers=int(config.get("evaluation", {}).get("num_workers", 0)),
        )

    model = Stage1CorrespondenceModel(config).to(args.device)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        missing, unexpected = model.load_compatible_checkpoint(payload["model"])
        print(
            f"initialized model weights from {args.init_checkpoint} "
            f"(new candidate-graph keys initialized={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )
    stage = str(config.get("training", {}).get("stage", "joint"))
    params = _configure_training_stage(model, stage)
    if not params:
        raise RuntimeError(f"No trainable parameters for training.stage={stage}.")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    use_amp = bool(config["training"].get("amp", True)) and str(args.device).startswith("cuda")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_amp)
    )
    start_epoch = 1
    global_step = 0
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        missing, _ = model.load_compatible_checkpoint(payload["model"])
        _configure_training_stage(model, stage)
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError:
            if not missing:
                raise
            # A pre-graph optimizer has fewer parameters. Its model weights are
            # usable, but its optimizer state cannot be mapped safely.
            print(
                "checkpoint predates candidate graph parameters; resumed model weights with a fresh optimizer",
                flush=True,
            )
            payload["epoch"] = 0
            payload["step"] = 0
            payload["completed_epoch"] = True
        if payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        restored_epoch = int(payload.get("epoch", 0))
        restored_completed_epoch = bool(payload.get("completed_epoch", True))
        start_epoch = restored_epoch + 1 if restored_completed_epoch else max(restored_epoch, 1)
        global_step = int(payload.get("step", 0))
        best_val_loss = payload.get("best_val_loss")
        best_val_loss = float(best_val_loss) if best_val_loss is not None else float("inf")
        best_val_metric = payload.get("best_val_metric")
        best_val_metric = float(best_val_metric) if best_val_metric is not None else float("inf")
    else:
        best_val_loss = float("inf")
        best_val_metric = float("inf")

    epochs = int(config["training"].get("epochs", 1))
    log_interval = int(config["training"].get("log_interval", 10))
    visualization_enabled = bool(config["visualization"].get("enabled", True))
    frames_per_scene = int(config["visualization"].get("frames_per_scene", 3))
    train_visualization_samples = _select_fixed_visualization_samples(dataset, frames_per_scene)
    val_visualization_samples = (
        _select_fixed_visualization_samples(val_dataset, frames_per_scene)
        if val_dataset is not None
        else []
    )
    latest_checkpoint_name = str(config["training"].get("checkpoint_name", "stage1_latest.pth"))
    legacy_best_name = str(config["training"].get("best_checkpoint_name", "stage1_best.pth"))
    best_loss_checkpoint_name = str(
        config["training"].get(
            "best_loss_checkpoint_name",
            _checkpoint_path_with_suffix(Path(legacy_best_name), "_loss"),
        )
    )
    best_metric_checkpoint_name = str(
        config["training"].get(
            "best_metric_checkpoint_name",
            _checkpoint_path_with_suffix(Path(legacy_best_name), "_metric"),
        )
    )
    latest_checkpoint_path = ckpt_dir / latest_checkpoint_name
    best_loss_checkpoint_template_path = ckpt_dir / best_loss_checkpoint_name
    best_metric_checkpoint_template_path = ckpt_dir / best_metric_checkpoint_name
    model.train()
    stop = False
    latest_checkpoint_completed = False
    try:
        with metrics_path.open("a", encoding="utf-8") as metric_handle:
            for epoch in range(start_epoch, epochs + 1):
                progress = tqdm(total=len(loader_iterable), desc=f"epoch {epoch}/{epochs}") if tqdm is not None else None
                running: dict[str, float] = defaultdict(float)
                count = 0
                for batch_payload, load_errors in loader_iterable:
                    if batch_payload is None:
                        if progress is not None:
                            progress.update(1)
                        continue
                    batch = move_batch_to_device(batch_payload, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        output = model(batch)
                        losses = compute_losses(output, config)
                    if not torch.isfinite(losses.total):
                        raise FloatingPointError(f"Non-finite loss at step {global_step}: {losses.total.item()}")
                    scaler.scale(losses.total).backward()
                    if float(config["training"].get("grad_clip_norm", 0.0)) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, float(config["training"]["grad_clip_norm"]))
                    scaler.step(optimizer)
                    scaler.update()

                    global_step += 1
                    count += 1
                    record = {name: _metric_to_float(value) for name, value in losses.metrics.items()}
                    record.update({"epoch": epoch, "step": global_step, "sample_ids": batch.sample_ids})
                    metric_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    metric_handle.flush()
                    for key, value in record.items():
                        if isinstance(value, float):
                            running[key] += value
                            writer.add_scalar(_tensorboard_scalar_tag(key), value, global_step)
                    if global_step % log_interval == 0:
                        head_loss_key = "loss_matchability" if stage == "coarse" else "loss_certainty"
                        msg = {
                            k: running[k] / max(count, 1)
                            for k in (
                                "loss_total",
                                "loss_coarse",
                                "loss_refine",
                                head_loss_key,
                                "coarse_top1_accuracy",
                                "refine_pixel_error",
                            )
                            if k in running
                        }
                        if progress is not None:
                            progress.set_postfix(
                                step=global_step,
                                total=f"{msg.get('loss_total', 0.0):.4f}",
                                coarse=f"{msg.get('loss_coarse', 0.0):.4f}",
                                refine=f"{msg.get('loss_refine', 0.0):.4f}",
                                head_loss=f"{msg.get(head_loss_key, 0.0):.4f}",
                                top1=f"{msg.get('coarse_top1_accuracy', 0.0):.4f}",
                                refine_px=f"{msg.get('refine_pixel_error', 0.0):.4f}",
                            )
                        else:
                            print(f"step={global_step} " + " ".join(f"{k}={v:.4f}" for k, v in msg.items()), flush=True)
                    if progress is not None:
                        progress.update(1)
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                        break
                if progress is not None:
                    progress.close()
                epoch_completed = not stop
                if visualization_enabled:
                    _save_fixed_visualizations(
                        model,
                        dataset,
                        train_visualization_samples,
                        config,
                        args.device,
                        use_amp,
                        vis_dir,
                        epoch,
                    )
                latest_checkpoint_completed = epoch_completed
                _save_versioned_checkpoint(
                    latest_checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    config,
                    keep_previous_versions=False,
                    completed_epoch=epoch_completed,
                    best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
                    best_val_metric=best_val_metric if best_val_metric != float("inf") else None,
                )
                if val_loader is not None and global_step > 0:
                    val_metrics = _run_validation(
                        model,
                        val_loader,
                        config,
                        args.device,
                        use_amp,
                        max_batches=args.max_val_batches,
                    )
                    if visualization_enabled and val_dataset is not None:
                        _save_fixed_visualizations(
                            model,
                            val_dataset,
                            val_visualization_samples,
                            config,
                            args.device,
                            use_amp,
                            val_vis_dir,
                            epoch,
                        )
                    val_loss = val_metrics.get("loss_total")
                    val_metric = val_metrics.get("mean_metric_error")
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"val/{_tensorboard_scalar_tag(key)}", value, global_step)

                    best_checkpoint_updated = False
                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_versioned_checkpoint(
                            best_loss_checkpoint_template_path,
                            model,
                            optimizer,
                            scaler,
                            epoch,
                            global_step,
                            config,
                            keep_previous_versions=True,
                            completed_epoch=epoch_completed,
                            best_val_loss=best_val_loss,
                            best_val_metric=best_val_metric if best_val_metric != float("inf") else None,
                        )
                        best_checkpoint_updated = True

                    if val_metric is not None and val_metric < best_val_metric:
                        best_val_metric = val_metric
                        _save_versioned_checkpoint(
                            best_metric_checkpoint_template_path,
                            model,
                            optimizer,
                            scaler,
                            epoch,
                            global_step,
                            config,
                            keep_previous_versions=True,
                            completed_epoch=epoch_completed,
                            best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
                            best_val_metric=best_val_metric,
                        )
                        best_checkpoint_updated = True

                    if best_checkpoint_updated:
                        _save_versioned_checkpoint(
                            latest_checkpoint_path,
                            model,
                            optimizer,
                            scaler,
                            epoch,
                            global_step,
                            config,
                            keep_previous_versions=False,
                            completed_epoch=epoch_completed,
                            best_val_loss=best_val_loss,
                            best_val_metric=best_val_metric,
                        )
                if stop:
                    break
    finally:
        if global_step > 0 and not latest_checkpoint_completed:
            _save_versioned_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scaler,
                max(start_epoch, epoch if "epoch" in locals() else start_epoch),
                global_step,
                config,
                keep_previous_versions=False,
                completed_epoch=False,
                best_val_loss=best_val_loss if best_val_loss != float("inf") else None,
                best_val_metric=best_val_metric if best_val_metric != float("inf") else None,
            )
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
