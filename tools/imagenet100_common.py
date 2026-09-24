"""Shared, offline-safe ImageNet-100 benchmark definition.

The selected subset is the PODNet/DER/DyTox-compatible first 100 WNIDs in
ImageNet's canonical sorted WNID order.  This module deliberately contains no
network or dataset-library imports so it is safe to use on a cluster node.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SEED = 42
NUM_CLASSES = 100
NUM_TASKS = 5
CLASSES_PER_TASK = 20
RANKEXT_SCHEDULE = [16, 32, 48, 64, 80]

# Reused from scripts/verify_imagenet100_local.py and the repository's prior
# ImageNet preparation report.  Do not silently replace this with the unrelated
# clane9/imagenet-100 class definition.
IMAGENET100_SYNSETS = [
    "n01440764", "n01443537", "n01484850", "n01491361", "n01494475",
    "n01496331", "n01498041", "n01514668", "n01514859", "n01518878",
    "n01530575", "n01531178", "n01532829", "n01534433", "n01537544",
    "n01558993", "n01560419", "n01580077", "n01582220", "n01592084",
    "n01601694", "n01608432", "n01614925", "n01616318", "n01622779",
    "n01629819", "n01630670", "n01631663", "n01632458", "n01632777",
    "n01641577", "n01644373", "n01644900", "n01664065", "n01665541",
    "n01667114", "n01667778", "n01669191", "n01675722", "n01677366",
    "n01682714", "n01685808", "n01687978", "n01688243", "n01689811",
    "n01692333", "n01693334", "n01694178", "n01695060", "n01697457",
    "n01698640", "n01704323", "n01728572", "n01728920", "n01729322",
    "n01729977", "n01734418", "n01735189", "n01737021", "n01739381",
    "n01740131", "n01742172", "n01744401", "n01748264", "n01749939",
    "n01751748", "n01753488", "n01755581", "n01756291", "n01768244",
    "n01770081", "n01770393", "n01773157", "n01773549", "n01773797",
    "n01774384", "n01774750", "n01775062", "n01776313", "n01784675",
    "n01795545", "n01796340", "n01797886", "n01798484", "n01806143",
    "n01806567", "n01807496", "n01817953", "n01818515", "n01819313",
    "n01820546", "n01824575", "n01828970", "n01829413", "n01833805",
    "n01843065", "n01843383", "n01847000", "n01855032", "n01855672",
]

# Exact public clane9/imagenet-100 label order.  The dataset card states that
# its integer labels follow the CMC ``imagenet100.txt`` order.  Keep this
# separate from IMAGENET100_SYNSETS: the public mirror is useful for source
# inspection, but it is not the canonical thesis subset used by the benchmark.
CLANE9_IMAGENET100_SYNSETS = [
    "n02869837", "n01749939", "n02488291", "n02107142", "n13037406",
    "n02091831", "n04517823", "n04589890", "n03062245", "n01773797",
    "n01735189", "n07831146", "n07753275", "n03085013", "n04485082",
    "n02105505", "n01983481", "n02788148", "n03530642", "n04435653",
    "n02086910", "n02859443", "n13040303", "n03594734", "n02085620",
    "n02099849", "n01558993", "n04493381", "n02109047", "n04111531",
    "n02877765", "n04429376", "n02009229", "n01978455", "n02106550",
    "n01820546", "n01692333", "n07714571", "n02974003", "n02114855",
    "n03785016", "n03764736", "n03775546", "n02087046", "n07836838",
    "n04099969", "n04592741", "n03891251", "n02701002", "n03379051",
    "n02259212", "n07715103", "n03947888", "n04026417", "n02326432",
    "n03637318", "n01980166", "n02113799", "n02086240", "n03903868",
    "n02483362", "n04127249", "n02089973", "n03017168", "n02093428",
    "n02804414", "n02396427", "n04418357", "n02172182", "n01729322",
    "n02113978", "n03787032", "n02089867", "n02119022", "n03777754",
    "n04238763", "n02231487", "n03032252", "n02138441", "n02104029",
    "n03837869", "n03494278", "n04136333", "n03794056", "n03492542",
    "n02018207", "n04067472", "n03930630", "n03584829", "n02123045",
    "n04229816", "n02100583", "n03642806", "n04336792", "n03259280",
    "n02116738", "n02108089", "n03424325", "n01855672", "n02090622",
]


def task_split(seed: int = SEED) -> dict[str, Any]:
    """Return the deterministic benchmark-local class/task mapping."""
    order = list(range(NUM_CLASSES))
    random.Random(seed).shuffle(order)
    tasks = []
    for task_index in range(NUM_TASKS):
        new_ids = list(range(task_index * CLASSES_PER_TASK, (task_index + 1) * CLASSES_PER_TASK))
        original_ids = [order[new_id] for new_id in new_ids]
        tasks.append(
            {
                "task_index": task_index,
                "task_number": task_index + 1,
                "new_class_ids": new_ids,
                "original_class_ids": original_ids,
                "synsets": [IMAGENET100_SYNSETS[i] for i in original_ids],
            }
        )
    return {
        "schema_version": 2,
        "dataset_identifier": "ImageNet-100/PODNet-DER-DyTox-first100-sorted-wnids",
        "seed": seed,
        "num_classes": NUM_CLASSES,
        "num_tasks": NUM_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "selected_synsets": IMAGENET100_SYNSETS,
        "original_id_to_new_id": {str(original): new for new, original in enumerate(order)},
        "new_id_to_original_id": {str(new): original for new, original in enumerate(order)},
        "tasks": tasks,
    }


def classes_metadata(display_names: list[str] | None = None) -> dict[str, Any]:
    names = display_names or IMAGENET100_SYNSETS
    if len(names) != NUM_CLASSES:
        raise ValueError("display_names must contain exactly 100 entries")
    split = task_split()
    return {
        "schema_version": 2,
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "selected_synsets": IMAGENET100_SYNSETS,
        "classes": [
            {
                "benchmark_id": i,
                "original_label_id": i,
                "synset": synset,
                "display_name": names[i],
            }
            for i, synset in enumerate(IMAGENET100_SYNSETS)
        ],
        "task_split": split,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_prepared_datasets(data_root: str | Path) -> dict[str, Any]:
    """Load prepared JPEGs through a local Dataset only; never resolve a Hub id."""
    from datasets import Dataset, Image as HFImage

    root = Path(data_root).resolve()
    manifest = json.loads((root / "metadata" / "dataset_manifest.json").read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "calibration", "test"):
        rows = sorted((row for row in manifest["samples"] if row["split"] == split), key=lambda row: row["relative_path"])
        ds = Dataset.from_dict({
            "image": [str(root / row["relative_path"]) for row in rows],
            "label": [int(row["benchmark_class_id"]) for row in rows],
            "sample_id": [str(row["sample_id"]) for row in rows],
        })
        result[split] = ds.cast_column("image", HFImage())
    return result
