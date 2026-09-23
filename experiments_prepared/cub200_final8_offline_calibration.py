#!/usr/bin/env python
"""CPU-only validation-selected calibration for the improved final CUB run.

This script intentionally knows nothing about CUB images, CLIP, datasets, or
training. It consumes only the saved ``validation.pt``, ``test.pt``, and
``classifier.pt`` artifacts emitted by the improved production script.

All task-block transforms are column-wise: every sample receives the same
calibration of each class block. True labels are used only to score restricted
accuracy, never to decide which logits to transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "simple_avg",
    "simple_avg_kd_oldseen_T2_warmup",
    "simple_avg_dense_orth_lam20",
    "simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup",
    "rank_extension",
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_normalized_lam50",
    "rank_extension_factor_orth_normalized_lam50_fullkd_T2_protect30",
]
SIMPLEAVG = set(METHODS[:4])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_split(path: Path) -> List[List[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = [list(map(int, row["new_class_ids"])) for row in data["tasks"]]
    assert len(tasks) == 5 and all(len(task) == 40 for task in tasks)
    return tasks


def load_pt(path: Path) -> Mapping:
    return torch.load(path, map_location="cpu")


def class_to_task(class_splits: Sequence[Sequence[int]]) -> Dict[int, int]:
    return {int(cls): task for task, classes in enumerate(class_splits) for cls in classes}


def task_ids(labels: torch.Tensor, lookup: Mapping[int, int]) -> torch.Tensor:
    return torch.tensor([lookup[int(label)] for label in labels.tolist()], dtype=torch.long)


def metrics(logits: torch.Tensor, labels: torch.Tensor, splits: Sequence[Sequence[int]], lookup: Mapping[int, int]) -> Dict[str, float]:
    tids = task_ids(labels, lookup)
    open_pred = logits.argmax(dim=1)
    restricted_pred = torch.empty_like(open_pred)
    predicted_task = torch.div(open_pred, 40, rounding_mode="floor")
    for task, classes in enumerate(splits):
        mask = tids == task
        indices = torch.tensor(classes, dtype=torch.long)
        restricted_pred[mask] = indices[logits[mask][:, indices].argmax(dim=1)]
    open_acc = float((open_pred == labels).float().mean().item() * 100.0)
    restricted_acc = float((restricted_pred == labels).float().mean().item() * 100.0)
    task_id_acc = float((predicted_task == tids).float().mean().item() * 100.0)
    return {
        "all_seen": open_acc,
        "restricted": restricted_acc,
        "gap": restricted_acc - open_acc,
        "task_id": task_id_acc,
    }


def apply_blocks(logits: torch.Tensor, splits: Sequence[Sequence[int]], scales=None, biases=None) -> torch.Tensor:
    out = logits.clone()
    for task, classes in enumerate(splits):
        idx = torch.tensor(classes, dtype=torch.long)
        if scales is not None:
            out[:, idx] *= float(scales[task])
        if biases is not None:
            out[:, idx] += float(biases[task])
    return out


def shrinkage_penalty(scales: Sequence[float], biases: Sequence[float]) -> float:
    # Small, fixed priors keep the five task parameters near the identity
    # transform without allowing test-driven selection or per-class fitting.
    return 0.01 * sum((float(scale) - 1.0) ** 2 for scale in scales) + 0.001 * sum(float(bias) ** 2 for bias in biases)


def validation_objective(logits: torch.Tensor, labels: torch.Tensor, splits, lookup, scales, biases) -> float:
    score = metrics(apply_blocks(logits, splits, scales, biases), labels, splits, lookup)["all_seen"]
    return score - shrinkage_penalty(scales, biases)


def fit_block_calibration(logits: torch.Tensor, labels: torch.Tensor, splits, lookup, mode: str) -> Tuple[List[float], List[float]]:
    scales = [1.0] * len(splits)
    biases = [0.0] * len(splits)
    scale_grid = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.80, 2.20]
    bias_grid = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    use_scale = mode in {"scale", "affine"}
    use_bias = mode in {"bias", "affine"}
    for _ in range(3):
        for task in range(len(splits)):
            best = (float("-inf"), scales[task], biases[task])
            scale_values = scale_grid if use_scale else [scales[task]]
            bias_values = bias_grid if use_bias else [biases[task]]
            for scale in scale_values:
                for bias in bias_values:
                    trial_scales = list(scales)
                    trial_biases = list(biases)
                    trial_scales[task] = scale
                    trial_biases[task] = bias
                    objective = validation_objective(logits, labels, splits, lookup, trial_scales, trial_biases)
                    if objective > best[0] + 1e-12:
                        best = (objective, scale, bias)
            scales[task], biases[task] = best[1], best[2]
    if not use_scale:
        scales = [1.0] * len(splits)
    if not use_bias:
        biases = [0.0] * len(splits)
    return scales, biases


def fit_weight_alignment(weights: torch.Tensor, splits: Sequence[Sequence[int]]) -> List[float]:
    norms = weights.float().norm(dim=1)
    scales = [1.0] * len(splits)
    for task in range(1, len(splits)):
        old_idx = torch.tensor([cls for prior in splits[:task] for cls in prior], dtype=torch.long)
        new_idx = torch.tensor(splits[task], dtype=torch.long)
        target = float(norms[old_idx].mean().item())
        current = max(float(norms[new_idx].mean().item()), 1e-8)
        scales[task] = target / current
    return scales


def candidate_logits(raw_logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, splits, lookup):
    candidates = {"original": (raw_logits, {"scales": [1.0] * 5, "biases": [0.0] * 5})}
    wa_scales = fit_weight_alignment(weights, splits)
    candidates["WA"] = (apply_blocks(raw_logits, splits, wa_scales, None), {"scales": wa_scales, "biases": [0.0] * 5})
    for mode, name in [("scale", "scale"), ("bias", "bias"), ("affine", "affine")]:
        scales, biases = fit_block_calibration(raw_logits, labels, splits, lookup, mode)
        candidates[name] = (apply_blocks(raw_logits, splits, scales, biases), {"scales": scales, "biases": biases})
    return candidates


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = (args.output_dir or artifact_root.parent / "offline_calibration").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "experiments_prepared" / "splits" / "cub200_class_order_seed42_5x40.json"
    splits = load_split(split_path)
    lookup = class_to_task(splits)

    rows = []
    report = [
        "# Final improved CUB-200 offline calibration",
        "",
        "CPU-only. No CLIP, dataset, image inference, CUDA, or training is used.",
        "Calibration parameters are fit on validation logits only; test labels are read only after selection.",
        "",
        "## Raw versus calibrated results",
        "",
        "| Method | Raw all-seen | Raw restricted | Selected | Calibrated all-seen | Calibrated restricted | Gain | Gap before | Gap after | Task-ID before | Task-ID after |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        method_dir = artifact_root / method
        val = load_pt(method_dir / "validation.pt")
        classifier = load_pt(method_dir / "classifier.pt")
        val_logits = val["logits"].float()
        val_labels = val["labels"].long()
        weights = classifier["weight"].float()
        assert val_logits.shape[1] == 200 and weights.shape == (200, 768)
        raw_val = metrics(val_logits, val_labels, splits, lookup)
        candidates_val = candidate_logits(val_logits, val_labels, weights, splits, lookup)
        guard = 1.0 if method in SIMPLEAVG else 2.0
        eligible = []
        for name, (candidate, params) in candidates_val.items():
            candidate_metrics = metrics(candidate, val_labels, splits, lookup)
            if candidate_metrics["restricted"] >= raw_val["restricted"] - guard:
                eligible.append((candidate_metrics["all_seen"], name, candidate_metrics, params))
        if not eligible:
            raise RuntimeError(f"No validation candidate passed the restricted guardrail for {method}")
        _, selected, selected_val_metrics, params = max(eligible, key=lambda row: (row[0], -shrinkage_penalty(row[3]["scales"], row[3]["biases"])))

        test = load_pt(method_dir / "test.pt")
        test_logits = test["logits"].float()
        test_labels = test["labels"].long()
        raw_test = metrics(test_logits, test_labels, splits, lookup)
        calibrated_test_logits = apply_blocks(test_logits, splits, params["scales"], params["biases"])
        calibrated_test = metrics(calibrated_test_logits, test_labels, splits, lookup)
        row = {
            "method": method,
            "raw_all_seen_pct": raw_test["all_seen"],
            "raw_restricted_pct": raw_test["restricted"],
            "raw_gap_pp": raw_test["gap"],
            "raw_task_id_pct": raw_test["task_id"],
            "selected_calibration": selected,
            "validation_selected_all_seen_pct": selected_val_metrics["all_seen"],
            "validation_selected_restricted_pct": selected_val_metrics["restricted"],
            "test_calibrated_all_seen_pct": calibrated_test["all_seen"],
            "test_calibrated_restricted_pct": calibrated_test["restricted"],
            "test_calibrated_gap_pp": calibrated_test["gap"],
            "test_calibrated_task_id_pct": calibrated_test["task_id"],
            "gain_pp": calibrated_test["all_seen"] - raw_test["all_seen"],
            "restricted_delta_pp": calibrated_test["restricted"] - raw_test["restricted"],
            "scales": json.dumps(params["scales"]),
            "biases": json.dumps(params["biases"]),
        }
        rows.append(row)
        report.append(
            f"| {method} | {raw_test['all_seen']:.3f} | {raw_test['restricted']:.3f} | {selected} | "
            f"{calibrated_test['all_seen']:.3f} | {calibrated_test['restricted']:.3f} | "
            f"{row['gain_pp']:+.3f} | {raw_test['gap']:.3f} | {calibrated_test['gap']:.3f} | "
            f"{raw_test['task_id']:.3f} | {calibrated_test['task_id']:.3f} |"
        )

    import csv
    csv_path = output_dir / "final8_offline_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report.extend([
        "",
        "## Selection rules",
        "",
        "- Scale, bias, and affine parameters are task-block parameters only.",
        "- Parameters are selected by validation all-seen accuracy with fixed shrinkage toward scale=1 and bias=0.",
        "- SimpleAvg restricted validation loss guardrail: 1 percentage point.",
        "- RankExt restricted validation loss guardrail: 2 percentage points.",
        "- WA is sequential old/new classifier-row norm alignment using saved classifier rows.",
        "- Test labels are not accessed until the selected transform is frozen.",
        "",
        "## Artifact requirements",
        "",
        f"- Methods evaluated: {len(rows)}/8.",
        "- Required files per method: validation logits, test logits, classifier weights/biases.",
        "- Features are not required by the selected calibration family and are therefore not loaded.",
    ])
    report_path = output_dir / "final8_offline_calibration_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"CPU ONLY: YES")
    print(f"METHODS: {len(rows)}/8")
    print(f"CSV: {csv_path}")
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
