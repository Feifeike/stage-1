from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from stage1.utils.config import load_config
from stage1.utils.checkpointing import (
    checkpoint_path_with_progress,
    sha256_file,
    update_checkpoint_alias,
)
from stage1.models.confidence import Stage1ConfidenceModel, compute_confidence_loss
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent Stage-1 confidence head.")
    parser.add_argument("--config", type=Path, default=Path("models/stage1/configs/config_confidence.yaml"))
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        required=True,
        help="Existing coarse+refine Stage-1 checkpoint; it is loaded and then frozen.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume a confidence checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Override the training manifest.")
    parser.add_argument("--val_manifest", type=Path, default=None, help="Override the validation manifest.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--overfit_batches", type=int, default=None)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def _metric_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _ema(previous: float | None, current: float, alpha: float) -> float:
    """Smooth a display-only scalar without changing the training value."""

    return current if previous is None else alpha * previous + (1.0 - alpha) * current


def _load_base_checkpoint(model: Stage1ConfidenceModel, path: Path, device: str) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.stage1.load_compatible_checkpoint(state)
    print(
        f"initialized frozen coarse+refine model from {path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    model.stage1.eval()
    model.stage1.requires_grad_(False)


def _load_confidence_checkpoint(
    model: Stage1ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    path: Path,
    device: str,
    base_checkpoint_sha256: str,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_base_hash = payload.get("base_checkpoint_sha256")
    if saved_base_hash is None:
        print(
            "warning: resumed confidence checkpoint has no base checkpoint hash; "
            "the Stage-1 pairing cannot be verified",
            flush=True,
        )
    elif saved_base_hash != base_checkpoint_sha256:
        raise ValueError(
            "The confidence checkpoint was trained with a different Stage-1 base checkpoint: "
            f"saved SHA-256={saved_base_hash}, requested SHA-256={base_checkpoint_sha256}."
        )
    head_state = payload.get("confidence_head")
    if head_state is None:
        state = payload.get("model", {})
        head_state = {
            key.removeprefix("confidence_head."): value
            for key, value in state.items()
            if key.startswith("confidence_head.")
        }
    missing, unexpected = model.confidence_head.load_state_dict(head_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"invalid confidence checkpoint: missing={missing}, unexpected={unexpected}")
    if payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0)), int(payload.get("step", 0)), float(payload.get("best_val_loss", float("inf")))


def _save_checkpoint(
    path: Path,
    model: Stage1ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "confidence_head": model.confidence_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
            "base_checkpoint": str(base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "training_stage": "confidence_head",
            "config": config,
        },
        path,
    )


def _save_versioned_checkpoint(
    path_template: Path,
    model: Stage1ConfidenceModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    best_val_loss: float,
    *,
    keep_previous_versions: bool,
) -> Path:
    path = checkpoint_path_with_progress(path_template, epoch, step)
    _save_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        epoch,
        step,
        config,
        base_checkpoint,
        base_checkpoint_sha256,
        best_val_loss,
    )
    update_checkpoint_alias(path_template, path, keep_previous_versions=keep_previous_versions)
    return path


@torch.no_grad()
def _run_validation(model, loader, config: dict, device: str, use_amp: bool, max_batches: int | None) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch_index, (batch_payload, _load_errors) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if batch_payload is None:
            continue
        batch = move_batch_to_device(batch_payload, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            output = model(batch)
            loss_output = compute_confidence_loss(output, batch, config)
        for key, value in loss_output.metrics.items():
            totals[key] += _metric_to_float(value)
        count += 1
    model.train()
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_cfg = config.setdefault("training", {})
    output_dir = Path(args.output_dir or training_cfg.get("output_dir", "outputs/stage1_confidence"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    latest_checkpoint_template = checkpoint_dir / str(
        training_cfg.get("checkpoint_name", "confidence_latest.pth")
    )
    best_checkpoint_template = checkpoint_dir / str(
        training_cfg.get("best_checkpoint_name", "confidence_best.pth")
    )
    metrics_path = output_dir / "metrics.jsonl"
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if SummaryWriter is not None else _NullSummaryWriter()

    train_manifest = args.manifest or config["data"].get("train_manifest")
    if not train_manifest:
        raise ValueError("A confidence training manifest with monocular depth is required.")
    dataset = Stage1Dataset(train_manifest)
    loader = build_dataloader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=args.overfit_batches is None,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    if args.overfit_batches is not None:
        cached_batches = []
        for batch_index, item in enumerate(loader):
            cached_batches.append(item)
            if batch_index + 1 >= args.overfit_batches:
                break
        loader_iterable = cached_batches
    else:
        loader_iterable = loader

    val_loader = None
    val_manifest = args.val_manifest or config.get("data", {}).get("val_manifest")
    if val_manifest and not args.skip_validation:
        val_dataset = Stage1Dataset(val_manifest)
        val_loader = build_dataloader(
            val_dataset,
            batch_size=int(config.get("evaluation", {}).get("batch_size", training_cfg.get("batch_size", 1))),
            shuffle=False,
            num_workers=int(config.get("evaluation", {}).get("num_workers", 0)),
        )

    model = Stage1ConfidenceModel(config).to(args.device)
    base_checkpoint_sha256 = sha256_file(args.base_checkpoint)
    _load_base_checkpoint(model, args.base_checkpoint, args.device)
    optimizer = torch.optim.AdamW(
        model.confidence_head.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training_cfg.get("amp", True)) and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    tensorboard_loss_ema_alpha = float(training_cfg.get("tensorboard_loss_ema_alpha", 0.98))
    if not 0.0 <= tensorboard_loss_ema_alpha < 1.0:
        raise ValueError("training.tensorboard_loss_ema_alpha must be in [0, 1).")
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    if args.checkpoint is not None:
        restored_epoch, restored_step, best_val_loss = _load_confidence_checkpoint(
            model,
            optimizer,
            scaler,
            args.checkpoint,
            args.device,
            base_checkpoint_sha256,
        )
        start_epoch = restored_epoch + 1
        global_step = restored_step

    model.train()
    epochs = int(training_cfg.get("epochs", 1))
    log_interval = int(training_cfg.get("log_interval", 10))
    max_steps = args.max_steps
    stop = False
    tensorboard_loss_ema: float | None = None
    try:
        with metrics_path.open("a", encoding="utf-8") as metric_handle:
            for epoch in range(start_epoch, epochs + 1):
                progress = tqdm(total=len(loader_iterable), desc=f"confidence epoch {epoch}/{epochs}") if tqdm else None
                running: dict[str, float] = defaultdict(float)
                batch_count = 0
                for batch_payload, _load_errors in loader_iterable:
                    if batch_payload is None:
                        if progress:
                            progress.update(1)
                        continue
                    batch = move_batch_to_device(batch_payload, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        output = model(batch)
                        loss_output = compute_confidence_loss(output, batch, config)
                    if not torch.isfinite(loss_output.loss):
                        raise FloatingPointError(f"Non-finite confidence loss at step {global_step}: {loss_output.loss.item()}")
                    scaler.scale(loss_output.loss).backward()
                    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.confidence_head.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    batch_count += 1
                    record = {key: _metric_to_float(value) for key, value in loss_output.metrics.items()}
                    record.update({"epoch": epoch, "step": global_step, "sample_ids": batch.sample_ids})
                    metric_handle.write(json.dumps(record) + "\n")
                    metric_handle.flush()
                    for key, value in record.items():
                        if isinstance(value, float):
                            running[key] += value
                    tensorboard_loss_ema = _ema(
                        tensorboard_loss_ema,
                        record["loss_confidence"],
                        tensorboard_loss_ema_alpha,
                    )
                    writer.add_scalar("loss/confidence", tensorboard_loss_ema, global_step)
                    writer.add_scalar(
                        "scale/confidence_scale",
                        record["confidence_scale_mean"],
                        global_step,
                    )
                    writer.add_scalar("scale/equal_scale", record["equal_scale_mean"], global_step)
                    if progress:
                        progress.update(1)
                        if global_step % log_interval == 0:
                            progress.set_postfix(
                                loss=f"{running['loss_confidence'] / max(batch_count, 1):.4f}",
                                anchors=f"{running['num_valid_anchors'] / max(batch_count, 1):.1f}",
                            )
                    if max_steps is not None and global_step >= max_steps:
                        stop = True
                        break
                if progress:
                    progress.close()

                val_metrics = _run_validation(
                    model,
                    val_loader,
                    config,
                    args.device,
                    use_amp,
                    args.max_val_batches,
                ) if val_loader is not None else {}
                if val_metrics:
                    val_record = {f"val_{key}": value for key, value in val_metrics.items()}
                    val_record.update({"epoch": epoch, "step": global_step})
                    metric_handle.write(json.dumps(val_record) + "\n")
                    metric_handle.flush()
                    writer.add_scalar(
                        "loss/val_confidence",
                        val_metrics["loss_confidence"],
                        global_step,
                    )
                    if val_metrics.get("loss_confidence", float("inf")) < best_val_loss:
                        best_val_loss = val_metrics["loss_confidence"]
                        _save_versioned_checkpoint(
                            best_checkpoint_template,
                            model,
                            optimizer,
                            scaler,
                            epoch,
                            global_step,
                            config,
                            args.base_checkpoint,
                            base_checkpoint_sha256,
                            best_val_loss,
                            keep_previous_versions=True,
                        )
                _save_versioned_checkpoint(
                    latest_checkpoint_template,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    config,
                    args.base_checkpoint,
                    base_checkpoint_sha256,
                    best_val_loss,
                    keep_previous_versions=False,
                )
                if stop:
                    break
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
