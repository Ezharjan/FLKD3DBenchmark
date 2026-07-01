"""Shared training/evaluation engine for the FL+KD 3D point-cloud classification benchmark.

Holds the training/evaluation plumbing used by every trainer, so the behaviour
is identical across them:

* setup_perf      - GPU backend flags (cuDNN autotuner; TF32 where supported).
* autocast_ctx    - bf16 mixed-precision context (no GradScaler).
* build_loader    - DataLoader construction.
* augment_points  - vectorised train-time augmentation (no in-place mutation).
* evaluate        - evaluation from a full confusion matrix
                    (overall instance accuracy + macro per-class accuracy).
* RNG / checkpoint helpers used for auto-resume.

Precision follows the GPU's capabilities.  GPUs without native bf16 run fp32
(with the cuDNN autotuner on): these PointNet++ models are gather/FPS-bound
rather than matmul-bound, so fp32 costs little and throughput instead comes from
running many independent jobs in parallel.  GPUs with native bf16 switch to bf16
autocast (+ TF32); bf16 keeps the fp32 exponent range so the distance/FPS/
ball-query math is stable and needs no gradient scaler (which also matches the
hand-written SCAFFOLD/FedDyn SGD).  Evaluation always runs in fp32.
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
from typing import Dict, Optional

import numpy as np
import torch

# Pure-numpy point ops (torch-free; shared with the dataset factory).
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.append(_BASE)
from data_utils.pc_ops import augment_batch_np  # noqa: E402


# ---------------------------------------------------------------------------
# Backend / device setup
# ---------------------------------------------------------------------------

def setup_perf(use_cpu: bool = False, deterministic: bool = False) -> None:
    """Enable throughput-friendly math backends (cuDNN autotuner; TF32 if supported).

    The TF32 flags are ignored on GPUs that lack it, so this is safe to call on
    any CUDA device."""
    if use_cpu or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True


def pick_device(use_cpu: bool = False) -> torch.device:
    if use_cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')


def autocast_ctx(device: torch.device, enabled: bool):
    """Return a bf16 autocast context on CUDA, else a no-op context."""
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def amp_default(args) -> bool:
    """bf16 autocast default: ON only for CUDA GPUs with native bf16 support.

    GPUs without native bf16 run fp32; for these gather/FPS-bound PointNet++
    models that is close to optimal, so throughput instead comes from running
    many independent jobs in parallel.  GPUs with native bf16 use it
    automatically.  Override with --no_amp (force fp32) anywhere."""
    if getattr(args, 'use_cpu', False) or getattr(args, 'no_amp', False):
        return False
    if not torch.cuda.is_available():
        return False
    # torch.cuda.is_bf16_supported() exists from torch 1.10; on older builds fall
    # back to fp32 instead of raising AttributeError.
    is_bf16 = getattr(torch.cuda, 'is_bf16_supported', None)
    return bool(is_bf16()) if callable(is_bf16) else False


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def default_num_workers() -> int:
    """Respect the process's CPU affinity/allocation; cap to avoid oversubscription."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 2
    return max(2, min(8, n))


def resolve_num_workers(args) -> int:
    nw = getattr(args, 'num_workers', None)
    if nw is None or nw < 0:
        return 0 if os.name == 'nt' else default_num_workers()
    return int(nw)


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker's NumPy/Python RNG reproducibly.

    ``torch.initial_seed()`` already folds in the worker id and the process-wide
    seed set by :func:`set_global_seed`, so worker-side augmentation/sampling is
    reproducible given ``(seed, num_workers)`` yet differs across workers.  With
    ``persistent_workers=True`` this runs once per worker and the RNG then keeps
    advancing across epochs (so augmentation stays diverse epoch-to-epoch)."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def augment_collate(batch):
    """Collate a list of ``(points[N, C], label)`` samples AND apply the
    train-time augmentation inside the DataLoader worker.

    Running augmentation here (instead of on the main thread, between batches)
    lets the CPU augmentation overlap GPU compute, which removes the per-batch
    GPU stall.  Returns ``points`` already transposed to ``(B, C, N)`` plus a
    LongTensor of labels.  Only xyz is augmented; normals are left untouched
    (see :func:`data_utils.pc_ops.augment_batch_np`)."""
    pts = np.stack([np.asarray(b[0], dtype=np.float32) for b in batch])  # (B, N, C)
    targets = torch.as_tensor([int(b[1]) for b in batch], dtype=torch.long)
    pts = augment_batch_np(pts)                                   # xyz-only, returns a copy
    points = torch.from_numpy(pts).transpose(2, 1).contiguous()  # (B, C, N)
    return points, targets


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int,
                 drop_last: bool = False, pin_memory: bool = True,
                 augment: bool = False):
    """Build the DataLoader.

    When ``augment=True`` the train-time augmentation is performed in the worker
    processes via :func:`augment_collate` (pipelined with GPU compute) and the
    batch is delivered as ``(B, C, N)``.  Evaluation/test loaders use
    ``augment=False`` (default) and yield the raw ``(B, N, C)`` batch, which
    :func:`evaluate` transposes internally -- so test data is never augmented."""
    persistent = num_workers > 0
    kwargs = dict(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    if num_workers > 0:
        kwargs['prefetch_factor'] = 4
        kwargs['worker_init_fn'] = seed_worker
    if augment:
        kwargs['collate_fn'] = augment_collate
    return torch.utils.data.DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Vectorised train-time augmentation (delegates to torch-free pc_ops)
# ---------------------------------------------------------------------------

def augment_points(points: torch.Tensor) -> torch.Tensor:
    """PointNet++ augmentation: dropout -> scale -> shift.

    Returns a (B, C, N) tensor ready for the network.  Operates on a copy so
    that cached dataset samples are never mutated in place.
    """
    pts = augment_batch_np(points.detach().cpu().numpy())
    return torch.from_numpy(pts).transpose(2, 1).contiguous()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, num_class: int, device: torch.device,
             amp: bool = False) -> Dict[str, object]:
    """Evaluate a classifier from a full confusion matrix.

    Returns dict with ``instance_acc`` (overall accuracy / OA), ``class_acc``
    (macro per-class recall over classes present in the test set / mAcc),
    ``confusion`` (rows=gt, cols=pred) and ``n_eval``.  The confusion matrix
    accumulates counts, so the result is independent of batch size (it does not
    average per-batch ratios).
    """
    model.eval()
    conf = torch.zeros(num_class * num_class, dtype=torch.long)
    for points, target in loader:
        # Datasets yield (B, N, C) with C in {3,6}; transpose to (B, C, N).
        if points.dim() == 3 and points.shape[-1] in (3, 6) and points.shape[1] not in (3, 6):
            points = points.transpose(2, 1)
        points = points.to(device, non_blocking=True)
        target = target.to(device).long().view(-1)
        with autocast_ctx(device, amp):
            pred, _ = model(points)
        pred_choice = pred.argmax(dim=1).view(-1)
        idx = (target * num_class + pred_choice).to('cpu')
        conf += torch.bincount(idx, minlength=num_class * num_class)
    conf = conf.reshape(num_class, num_class)
    total = int(conf.sum().item())
    correct = int(conf.diag().sum().item())
    instance_acc = correct / max(1, total)

    row_sum = conf.sum(dim=1)
    present = row_sum > 0
    if present.any():
        recall = conf.diag()[present].double() / row_sum[present].double()
        class_acc = float(recall.mean().item())
    else:
        class_acc = 0.0
    return {
        'instance_acc': float(instance_acc),
        'class_acc': float(class_acc),
        'confusion': conf.numpy(),
        'n_eval': total,
    }


# ---------------------------------------------------------------------------
# RNG state (for exact-ish auto-resume)
# ---------------------------------------------------------------------------

def get_rng_state() -> Dict[str, object]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    try:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        t = state['torch']
        torch.set_rng_state(t.cpu() if hasattr(t, 'cpu') else t)
        if state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])
    except Exception as exc:  # pragma: no cover
        print(f'[engine] warning: could not fully restore RNG state: {exc}')


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------

def atomic_save(obj, path) -> None:
    """Write a checkpoint by writing to a temp file then renaming, so an
    interrupted write does not leave a partial file in place."""
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
