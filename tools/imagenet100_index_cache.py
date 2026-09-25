"""Fast, order-preserving ImageNet-100 class/task index construction.

The production benchmark loads image paths into HuggingFace ``Dataset``
objects.  This module only consumes the lightweight label column; it never
touches the image column.  The resulting integer indices are suitable for
``Dataset.select`` and preserve the order that ``Dataset.filter`` would have
returned for the same class predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def _class_key(class_ids: Iterable[int]) -> frozenset[int]:
    return frozenset(int(class_id) for class_id in class_ids)


@dataclass(frozen=True)
class ImageNetTaskIndexCache:
    """All fixed-protocol selections for one dataset split.

    ``class_indices`` and ``selection_indices`` are built once.  Lookup never
    runs a Python predicate over the dataset.  The fallback combines already
    cached per-class arrays and sorts their row numbers, so it still does not
    rescan the full label array; production calls use the precomputed keys.
    """

    labels: np.ndarray
    class_indices: Mapping[int, np.ndarray]
    selection_indices: Mapping[frozenset[int], np.ndarray]
    task_indices: tuple[np.ndarray, ...]
    seen_indices: tuple[np.ndarray, ...]
    old_indices: tuple[np.ndarray, ...]
    current_indices: tuple[np.ndarray, ...]

    @classmethod
    def build(
        cls,
        labels: Sequence[int] | np.ndarray,
        task_class_splits: Sequence[Sequence[int]],
    ) -> "ImageNetTaskIndexCache":
        label_array = np.asarray(labels, dtype=np.int64)
        if label_array.ndim != 1:
            raise ValueError(f"expected 1-D labels, got shape {label_array.shape}")

        class_ids = sorted({int(label) for split in task_class_splits for label in split})
        class_indices = {
            class_id: np.flatnonzero(label_array == class_id).astype(np.int64, copy=False)
            for class_id in class_ids
        }

        task_class_sets = [_class_key(split) for split in task_class_splits]
        seen_class_sets = [
            _class_key(class_id for split in task_class_splits[: task_index + 1] for class_id in split)
            for task_index in range(len(task_class_splits))
        ]
        old_class_sets = [
            _class_key(class_id for split in task_class_splits[:task_index] for class_id in split)
            for task_index in range(len(task_class_splits))
        ]
        current_class_sets = list(task_class_sets)

        required_sets = set(task_class_sets + seen_class_sets + old_class_sets)
        required_sets.update(_class_key([class_id]) for class_id in class_indices)
        # The three aggregate evaluation sets are selected repeatedly by the
        # production script and are therefore explicitly cached too.
        required_sets.update({
            _class_key(task_class_splits[0]),
            _class_key(class_id for split in task_class_splits[1:] for class_id in split),
            _class_key(class_id for split in task_class_splits for class_id in split),
        })

        selection_indices = {
            key: np.flatnonzero(np.isin(label_array, np.fromiter(sorted(key), dtype=np.int64)))
            .astype(np.int64, copy=False)
            for key in required_sets
        }

        return cls(
            labels=label_array,
            class_indices=class_indices,
            selection_indices=selection_indices,
            task_indices=tuple(selection_indices[key] for key in task_class_sets),
            seen_indices=tuple(selection_indices[key] for key in seen_class_sets),
            old_indices=tuple(selection_indices[key] for key in old_class_sets),
            current_indices=tuple(selection_indices[key] for key in current_class_sets),
        )

    def indices_for_classes(self, class_ids: Iterable[int]) -> np.ndarray:
        """Return source-order row indices for a class set."""
        key = _class_key(class_ids)
        if key in self.selection_indices:
            return self.selection_indices[key]
        if not key:
            return np.empty(0, dtype=np.int64)
        unknown = sorted(key.difference(self.class_indices))
        if unknown:
            raise KeyError(f"class IDs were not present in the cache: {unknown}")
        return np.sort(np.concatenate([self.class_indices[class_id] for class_id in sorted(key)]))

    def count_for_class(self, class_id: int) -> int:
        return int(self.class_indices[int(class_id)].size)


def select_dataset(dataset, indices: np.ndarray):
    """Select rows without evaluating a Python predicate over the dataset."""
    return dataset.select(np.asarray(indices, dtype=np.int64).tolist())


__all__ = ["ImageNetTaskIndexCache", "select_dataset"]
