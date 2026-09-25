"""Synthetic regression test for filter-to-index selection equivalence."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from datasets import Dataset

from imagenet100_index_cache import ImageNetTaskIndexCache, select_dataset


def _old_filter(dataset: Dataset, class_ids: list[int]) -> Dataset:
    allowed = set(class_ids)
    return dataset.filter(lambda row: int(row["label"]) in allowed)


def _assert_same_selection(dataset: Dataset, cache: ImageNetTaskIndexCache, class_ids: list[int]) -> None:
    old = _old_filter(dataset, class_ids)
    new = select_dataset(dataset, cache.indices_for_classes(class_ids))
    assert old.num_rows == new.num_rows
    assert old["sample_id"] == new["sample_id"]
    assert old["label"] == new["label"]


def test_filter_to_index_equivalence() -> None:
    labels = np.asarray([3, 0, 2, 3, 1, 4, 2, 0, 4, 1, 3, 2], dtype=np.int64)
    dataset = Dataset.from_dict({
        "sample_id": [f"sample-{i}" for i in range(labels.size)],
        "label": labels.tolist(),
        "image": [f"image-{i}.jpg" for i in range(labels.size)],
    })
    task_splits = [[0, 1], [2, 3], [4]]
    cache = ImageNetTaskIndexCache.build(labels, task_splits)

    for task_index, classes in enumerate(task_splits):
        _assert_same_selection(dataset, cache, classes)
        _assert_same_selection(dataset, cache, [class_id for split in task_splits[: task_index + 1] for class_id in split])
        _assert_same_selection(dataset, cache, [class_id for split in task_splits[:task_index] for class_id in split])

    _assert_same_selection(dataset, cache, [0, 1, 2, 3, 4])
    _assert_same_selection(dataset, cache, [2, 3, 4])


def test_production_path_has_no_dataset_filter_call() -> None:
    production = Path(__file__).resolve().parents[1] / (
        "experiments_prepared/final_8method_imagenet100_5x20_job4971615_canonical_ep9.py"
    )
    tree = ast.parse(production.read_text(encoding="utf-8"))
    filter_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "filter"
    ]
    assert not filter_calls, f"production Dataset.filter calls remain at {[node.lineno for node in filter_calls]}"


if __name__ == "__main__":
    test_filter_to_index_equivalence()
    test_production_path_has_no_dataset_filter_call()
    print("FILTER-TO-INDEX SEMANTIC EQUIVALENCE: PASS")
    print("PRODUCTION FULL TRAIN FILTER AUDIT: PASS")
