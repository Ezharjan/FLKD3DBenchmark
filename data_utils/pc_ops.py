"""Pure-NumPy point-cloud operations shared across the codebase.

This module deliberately has **no torch / no project imports** so the core
data logic (augmentation, sampling, stratified splitting, singleton filtering)
can be unit-tested in isolation and reused identically by the dataset factory
and the training engine.  All functions are deterministic given NumPy's global
RNG state (seed via ``numpy.random.seed``).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Train-time augmentation (vectorised; matches provider.py semantics)
# ---------------------------------------------------------------------------

def random_point_dropout(batch: np.ndarray, max_ratio: float = 0.875) -> np.ndarray:
    """Per-cloud random dropout; dropped points are set to the cloud's first point."""
    B, N = batch.shape[0], batch.shape[1]
    ratios = np.random.random(B) * max_ratio
    drop = np.random.random((B, N)) <= ratios[:, None]
    first = batch[:, 0:1, :]
    return np.where(drop[:, :, None], first, batch)


def random_scale_xyz(xyz: np.ndarray, low: float = 0.8, high: float = 1.25) -> np.ndarray:
    scales = np.random.uniform(low, high, xyz.shape[0])
    return xyz * scales[:, None, None]


def random_shift_xyz(xyz: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
    shifts = np.random.uniform(-shift_range, shift_range, (xyz.shape[0], 3))
    return xyz + shifts[:, None, :]


def augment_batch_np(points: np.ndarray) -> np.ndarray:
    """Apply dropout -> scale -> shift to a ``(B, N, C)`` batch (returns a copy)."""
    pts = np.array(points, dtype=np.float32, copy=True)
    pts = random_point_dropout(pts)
    pts[:, :, 0:3] = random_scale_xyz(pts[:, :, 0:3])
    pts[:, :, 0:3] = random_shift_xyz(pts[:, :, 0:3])
    return pts


# ---------------------------------------------------------------------------
# Channel handling & sampling
# ---------------------------------------------------------------------------

def coerce_channels(pts: np.ndarray, use_normals: bool) -> np.ndarray:
    """Return exactly 3 (xyz) or 6 (xyz+normal) channels, padding/truncating."""
    if use_normals:
        if pts.shape[1] >= 6:
            return pts[:, 0:6]
        base = pts[:, 0:3] if pts.shape[1] >= 3 else pts
        pad = np.zeros((pts.shape[0], 6 - base.shape[1]), dtype=pts.dtype)
        return np.concatenate([base, pad], axis=1)[:, 0:6]
    return pts[:, 0:3]


def farthest_point_sample_np(points: np.ndarray, npoint: int) -> np.ndarray:
    """Deterministic (given RNG) farthest-point sampling on xyz; returns sampled rows."""
    N = points.shape[0]
    xyz = points[:, :3]
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return points[centroids]


def sample_points(points: np.ndarray, npoints: int, use_fps: bool = False) -> np.ndarray:
    """Resample a cloud to exactly ``npoints`` rows (FPS, random subsample, or pad)."""
    if points.shape[0] == 0:
        raise RuntimeError("Point cloud is empty")
    if points.shape[0] == npoints:
        return points
    if points.shape[0] > npoints:
        if use_fps:
            return farthest_point_sample_np(points, npoints)
        idx = np.random.choice(points.shape[0], npoints, replace=False)
        return points[idx]
    reps = npoints - points.shape[0]
    extra_idx = np.random.choice(points.shape[0], reps, replace=True)
    return np.concatenate([points, points[extra_idx]], axis=0)


# ---------------------------------------------------------------------------
# Class filtering & stratified splitting
# ---------------------------------------------------------------------------

def filter_and_relabel(labels: Sequence[int], num_classes: int,
                       min_k: int = 1) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """Drop classes with fewer than ``min_k`` samples and compact the labels.

    Returns ``(keep_mask, remap, kept_classes)`` where ``keep_mask`` selects the
    samples to retain, ``remap`` maps original->new contiguous class ids, and
    ``kept_classes`` is the ordered list of retained original class ids.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    kept_classes = [c for c in range(num_classes) if counts[c] >= min_k]
    if not kept_classes:
        raise RuntimeError(f"No class has >= min_k={min_k} samples.")
    remap = {c: i for i, c in enumerate(kept_classes)}
    keep_mask = np.array([l in remap for l in labels], dtype=bool)
    return keep_mask, remap, kept_classes


def stratified_split(labels: Sequence[int], train_split: float,
                     seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class deterministic split.

    Every class with >= 2 samples contributes >= 1 sample to *both* train and
    test.  Singleton classes are kept in train only (cannot be tested).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.append(int(idx[0]))
            continue
        n_train = int(round(train_split * n))
        n_train = min(max(n_train, 1), n - 1)
        train_idx.extend(int(i) for i in idx[:n_train])
        test_idx.extend(int(i) for i in idx[n_train:])
    return (np.array(sorted(train_idx), dtype=np.int64),
            np.array(sorted(test_idx), dtype=np.int64))
