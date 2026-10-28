from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from stage1.utils.config import load_config
from stage1.utils.checkpointing import (
    checkpoint_path_with_progress,
    sha256_file,
    update_checkpoint_alias,
)
from stage1.models.confidence import Stage1ConfidenceModel, compute_confidence_loss
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.utils.distributed import (
    DistributedContext,
    all_ranks_have_batch,
    all_ranks_are_true,
    average_scalar_metrics,
    barrier,
    cleanup_distributed,
    initialize_distributed,
)

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
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
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


def _load_base_checkpoint(
    model: Stage1ConfidenceModel,
    path: Path,
    device: str,
    *,
    verbose: bool = True,
) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.stage1.load_compatible_checkpoint(state)
    if verbose:
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
    *,
    verbose: bool = True,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_base_hash = payload.get("base_checkpoint_sha256")
    if saved_base_hash is None:
        if verbose:
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
    distributed = initialize_distributed(args.device, args.local_rank)
    args.device = str(distributed.device)
    try:
        _run_training(args, distributed)
    finally:
        cleanup_distributed(distributed)


def _run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:
    config = load_config(args.config)
    training_cfg = config.setdefault("training", {})
    output_dir = Path(args.output_dir or training_cfg.get("output_dir", "outputs/stage1_confidence"))
    if distributed.is_main:
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
    if distributed.is_main:
        writer = SummaryWriter(log_dir=str(tensorboard_dir)) if SummaryWriter is not None else _NullSummaryWriter()
    else:
        writer = _NullSummaryWriter()

    train_manifest = args.manifest or config["data"].get("train_manifest")
    if not train_manifest:
        raise ValueError("A confidence training manifest with monocular depth is required.")
    dataset = Stage1Dataset(train_manifest)
    shuffle = args.overfit_batches is None
    train_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
        )
        if distributed.enabled and args.overfit_batches is None
        else None
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=shuffle if train_sampler is None else False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        sampler=train_sampler,
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
    val_dataset = None
    if val_manifest and not args.skip_validation and distributed.is_main:
        val_dataset = Stage1Dataset(val_manifest)
        val_loader = build_dataloader(
            val_dataset,
            batch_size=int(config.get("evaluation", {}).get("batch_size", training_cfg.get("batch_size", 1))),
            shuffle=False,
            num_workers=int(config.get("evaluation", {}).get("num_workers", 0)),
        )

    model = Stage1ConfidenceModel(config).to(args.device)
    base_checkpoint_sha256 = sha256_file(args.base_checkpoint)
    _load_base_checkpoint(model, args.base_checkpoint, args.device, verbose=distributed.is_main)
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
            verbose=distributed.is_main,
        )
        start_epoch = restored_epoch + 1
        global_step = restored_step

    train_model = (
        DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank] if distributed.device.type == "cuda" else None,
            output_device=distributed.local_rank if distributed.device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if distributed.enabled
        else model
    )

    model.train()
    epochs = int(training_cfg.get("epochs", 1))
    log_interval = int(training_cfg.get("log_interval", 10))
    max_steps = args.max_steps
    stop = False
    tensorboard_loss_ema: float | None = None
    try:
        metric_context = metrics_path.open("a", encoding="utf-8") if distributed.is_main else nullcontext(None)
        with metric_context as metric_handle:
            for epoch in range(start_epoch, epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                progress = (
                    tqdm(total=len(loader_iterable), desc=f"confidence epoch {epoch}/{epochs}")
                    if tqdm and distributed.is_main
                    else None
                )
                running: dict[str, float] = defaultdict(float)
                batch_count = 0
                for batch_payload, _load_errors in loader_iterable:
                    if not all_ranks_have_batch(batch_payload is not None, distributed):
                        if progress:
                            progress.update(1)
                        continue
                    batch = move_batch_to_device(batch_payload, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        output = train_model(batch)
                        loss_output = compute_confidence_loss(output, batch, config)
                    if not all_ranks_are_true(bool(torch.isfinite(loss_output.loss).item()), distributed):
                        raise FloatingPointError(
                            "Non-finite confidence loss on at least one worker at "
                            f"step {global_step}; this worker loss={loss_output.loss.item()}"
                        )
                    scaler.scale(loss_output.loss).backward()
                    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.confidence_head.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    batch_count += 1
                    record = average_scalar_metrics(loss_output.metrics, distributed)
                    record.update({
                        "epoch": epoch,
                        "step": global_step,
                        "sample_ids": batch.sample_ids if distributed.is_main else [],
                    })
                    if distributed.is_main:
                        metric_handle.write(json.dumps(record) + "\n")
                        metric_handle.flush()
                    for key, value in record.items():
                        if isinstance(value, float) and distributed.is_main:
                            running[key] += value
                    if distributed.is_main:
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

                barrier(distributed)
                if distributed.is_main:
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
                barrier(distributed)
                if stop:
                    break
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
