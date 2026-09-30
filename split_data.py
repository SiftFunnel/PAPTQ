"""Reproducible index-only dataset splitting; no private images are bundled.

The historical FaceScrub loader used a global seed-624 80/20 shuffle. Palm
and Iris experiments consumed pre-split NPZ files; their original raw-image
indices are not recoverable from this repository. The classwise function
below creates a new split and must not be called the historical split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray
    auxiliary: np.ndarray


def legacy_facescrub_indices(count: int, train_fraction: float = 0.8,
                             seed: int = 624) -> tuple[np.ndarray, np.ndarray]:
    """Match the legacy global shuffle and integer cutoff exactly."""
    if count < 2 or not 0 < train_fraction < 1:
        raise ValueError("count must exceed one and fraction must be in (0,1)")
    indices = np.arange(int(count), dtype=np.int64)
    np.random.RandomState(seed).shuffle(indices)
    cut = int(train_fraction * count)
    return indices[:cut], indices[cut:]


def classwise_target_auxiliary_indices(labels, target_classes, test_per_class: int,
                                       seed: int = 1) -> SplitIndices:
    """Make a new deterministic target train/test and disjoint-class auxiliary split.

    The target labels must be zero-based contiguous for a classifier head. The
    auxiliary indices are returned separately and never enter target PTQ.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must be a one-dimensional integer array")
    target_classes = tuple(int(x) for x in target_classes)
    if target_classes != tuple(range(len(target_classes))) or test_per_class < 1:
        raise ValueError("target classes must be 0..C-1; test_per_class >= 1")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in target_classes:
        indices = np.flatnonzero(labels == cls)
        if len(indices) <= test_per_class:
            raise ValueError(f"class {cls} has insufficient images")
        shuffled = rng.permutation(indices)
        test.extend(shuffled[:test_per_class].tolist())
        train.extend(shuffled[test_per_class:].tolist())
    auxiliary = np.flatnonzero(~np.isin(labels, target_classes)).astype(np.int64)
    return SplitIndices(np.sort(train).astype(np.int64),
                        np.sort(test).astype(np.int64), auxiliary)


def exact_image_overlap(images, first_indices, second_indices) -> set[str]:
    """Audit exact pixel duplicates across two index sets without exporting pixels."""
    images = np.asarray(images)
    def digests(indices):
        return {hashlib.sha256(np.ascontiguousarray(images[int(i)]).tobytes()).hexdigest()
                for i in indices}
    return digests(first_indices) & digests(second_indices)


def save_split_npz(path, images, labels, split: SplitIndices) -> None:
    """Write a user-owned NPZ in the format consumed by public_data.py.

    Refuses overwrite and refuses exact train/test image overlap. Auxiliary
    images are included only for compatibility, never used by target PTQ.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    images, labels = np.asarray(images), np.asarray(labels)
    if images.dtype != np.uint8 or len(images) != len(labels):
        raise ValueError("expected aligned uint8 images and labels")
    if exact_image_overlap(images, split.train, split.test):
        raise ValueError("target train/test contain exact duplicate images")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        train_data=images[split.train], train_labels=labels[split.train],
                        test_data=images[split.test], test_labels=labels[split.test],
                        auxiliary_data=images[split.auxiliary],
                        auxiliary_labels=labels[split.auxiliary])
