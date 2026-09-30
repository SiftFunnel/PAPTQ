"""Data adapters for user-supplied NPZ files; no images are distributed here.

The target-training split is the only split used for Fisher estimation,
activation calibration, and Stage-3 refinement. The test split is loaded for
the historical evaluation/checkpoint-selection protocol, never for updates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class _TargetNPZ(Dataset):
    filename: str

    def __init__(self, root, train=True, transform=None, target_transform=None, **_):
        path = Path(root).expanduser() / self.filename
        if not path.is_file():
            raise FileNotFoundError(f"Supply the dataset NPZ at {path}")
        prefix = "train" if train else "test"
        with np.load(path, allow_pickle=False) as archive:
            required = {f"{prefix}_data", f"{prefix}_labels"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path.name} misses {sorted(missing)}")
            self.data = np.asarray(archive[f"{prefix}_data"])
            self.labels = np.asarray(archive[f"{prefix}_labels"], dtype=np.int64)
        if len(self.data) != len(self.labels) or not len(self.data):
            raise ValueError(f"{prefix} images/labels have invalid lengths")
        if self.data.dtype != np.uint8:
            raise TypeError("Expected uint8 images; do not silently rescale source data")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class Palm(_TargetNPZ):
    filename = "palm.npz"


class IRIS_thousand(_TargetNPZ):
    filename = "iris_thousand.npz"


class FaceScrub(Dataset):
    """Legacy 80/20 index split after seed-624 global shuffle.

    This matches the original loader rule, not a class-stratified re-split.
    Callers supplying an already split NPZ should instead use _TargetNPZ.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 allocation=0.8, test_free=True, **_):
        path = Path(root).expanduser() / "facescrub_color1.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Supply the dataset NPZ at {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"actor_images", "actor_labels", "actress_images", "actress_labels"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path.name} misses {sorted(missing)}")
            images = np.concatenate((archive["actor_images"], archive["actress_images"]))
            labels = np.concatenate((archive["actor_labels"], archive["actress_labels"]))
        if images.dtype != np.uint8 or len(images) != len(labels):
            raise ValueError("FaceScrub arrays must be aligned uint8 images and labels")
        if test_free:
            # Preserve RandomState/np.random.shuffle behavior used by the old loader.
            order = np.arange(len(labels))
            np.random.RandomState(624).shuffle(order)
            images, labels = images[order], labels[order]
        split = int(float(allocation) * len(labels))
        portion = slice(0, split) if train else slice(split, None)
        self.data, self.labels = images[portion], labels[portion].astype(np.int64)
        self.transform, self.target_transform = transform, target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
