from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_name: str, local_rank_override: int | None = None) -> DistributedContext:
    """Initialize torchrun workers and select each worker's local device."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank_override if local_rank_override is not None else 0))
    requested_device = torch.device(device_name)
    enabled = world_size > 1

    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch process has no available CUDA device.")
        device_index = local_rank if enabled else requested_device.index
        if device_index is None:
            device_index = 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={device_index} is outside the visible CUDA device range "
                f"[0, {torch.cuda.device_count()}). Check CUDA_VISIBLE_DEVICES and --nproc_per_node."
            )
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = requested_device

    if enabled:
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not include torch.distributed support.")
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_ranks_have_batch(local_batch_is_valid: bool, context: DistributedContext) -> bool:
    """Keep ranks in lockstep when a complete local batch failed to load."""

    return all_ranks_are_true(local_batch_is_valid, context)


def all_ranks_are_true(local_condition: bool, context: DistributedContext) -> bool:
    """Return whether every worker reported a true condition."""

    if not context.enabled:
        return local_condition
    flag = torch.tensor(
        1 if local_condition else 0,
        dtype=torch.int32,
        device=context.device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def average_scalar_metrics(metrics: dict[str, torch.Tensor | float], context: DistributedContext) -> dict[str, float]:
    """Average scalar metrics across workers for consistent rank-zero logs."""

    names = sorted(metrics)
    if not names:
        return {}
    values = torch.stack(
        [
            torch.as_tensor(metrics[name], dtype=torch.float64, device=context.device).detach().reshape(())
            for name in names
        ]
    )
    if context.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return {name: float(value.item()) for name, value in zip(names, values)}


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()
