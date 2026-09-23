#!/usr/bin/env python3
"""CPU-only validation-fitted calibration for the full normalized Combined run.

This script reads only saved validation/test logits. It never imports CLIP,
loads a checkpoint, runs image inference, or reads test labels for fitting or
selection. The selected method is determined from validation all-seen/open
accuracy with the restricted-validation guardrail, then frozen for test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "R8" / "performance_improvement_research" / "cifar_job4971615_normalized_combined_full_seed42_9ep"
REPORT_PATH = ROOT / "R8" / "performance_improvement_research" / "cifar_job4971615_normalized_combined_full_report.md"
SHRINKAGE = 0.01
CLASS_SPLITS = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]


def restricted_accuracy(logits, labels):
    task_for_label = {c: i for i, block in enumerate(CLASS_SPLITS) for c in block}
    pred = []
    for row, label in zip(logits, labels):
        block = CLASS_SPLITS[task_for_label[int(label)]]
        masked = np.full(100, -np.inf, dtype=np.float64)
        masked[block] = row[block]
        pred.append(int(np.argmax(masked)))
    return float(np.mean(np.asarray(pred) == labels))


def open_accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=1) == labels))


def log_softmax_nll(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return float((-z[np.arange(len(labels)), labels] + np.log(np.exp(z).sum(axis=1))).mean())


def apply_params(logits, method, params):
    out = np.asarray(logits, dtype=np.float64).copy()
    for block, values in zip(CLASS_SPLITS, params):
        if method == "scale":
            out[:, block] *= math.exp(float(values[0]))
        elif method == "bias":
            out[:, block] += float(values[0])
        elif method == "affine":
            out[:, block] = out[:, block] * math.exp(float(values[0])) + float(values[1])
    return out


def golden_minimize(objective, lo, hi, iterations=48):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(iterations):
        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
        if objective(c) <= objective(d):
            hi = d
        else:
            lo = c
    return float((lo + hi) / 2.0)


def fit_scale(val_logits, val_labels):
    params = []
    for block in CLASS_SPLITS:
        mask = np.isin(val_labels, block)
        x = val_logits[mask]
        y = val_labels[mask]
        def objective(log_scale):
            z = x.copy()
            z[:, block] *= math.exp(float(log_scale))
            return log_softmax_nll(z, y) + SHRINKAGE * float(log_scale) ** 2
        params.append((golden_minimize(objective, -5.0, 5.0),))
    return params


def fit_bias(val_logits, val_labels):
    params = []
    for block in CLASS_SPLITS:
        mask = np.isin(val_labels, block)
        x = val_logits[mask]
        y = val_labels[mask]
        def objective(bias):
            z = x.copy()
            z[:, block] += float(bias)
            return log_softmax_nll(z, y) + SHRINKAGE * float(bias) ** 2
        params.append((golden_minimize(objective, -5.0, 5.0),))
    return params


def fit_affine(val_logits, val_labels):
    params = [(0.0, 0.0) for _ in CLASS_SPLITS]
    for _ in range(12):
        for block_index, block in enumerate(CLASS_SPLITS):
            mask = np.isin(val_labels, block)
            x = val_logits[mask]
            y = val_labels[mask]
            current_scale, current_bias = params[block_index]
            def scale_objective(log_scale):
                z = x.copy()
                z[:, block] = z[:, block] * math.exp(float(log_scale)) + current_bias
                return log_softmax_nll(z, y) + SHRINKAGE * float(log_scale) ** 2 + SHRINKAGE * current_bias ** 2
            current_scale = golden_minimize(scale_objective, -5.0, 5.0, iterations=32)
            def bias_objective(bias):
                z = x.copy()
                z[:, block] = z[:, block] * math.exp(current_scale) + float(bias)
                return log_softmax_nll(z, y) + SHRINKAGE * current_scale ** 2 + SHRINKAGE * float(bias) ** 2
            current_bias = golden_minimize(bias_objective, -5.0, 5.0, iterations=32)
            params[block_index] = (current_scale, current_bias)
    return params


def evaluate_candidate(name, val_logits, val_labels, params):
    logits = val_logits if name == "original" else apply_params(val_logits, name, params)
    return {
        "method": name,
        "val_open": open_accuracy(logits, val_labels),
        "val_restricted": restricted_accuracy(logits, val_labels),
        "val_nll": log_softmax_nll(logits, val_labels),
        "params": params,
    }


def load_trajectory(run_dir):
    path = run_dir / "tables" / "cl_trajectory.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def trajectory_summary(rows):
    result = []
    for task in range(1, 6):
        current = [r for r in rows if int(r["evaluated_task_step"]) == task and r["is_current_task"] == "True"]
        final = [r for r in rows if int(r["evaluated_task_step"]) == task and int(r["after_training_step"]) == 5]
        acquisition = current[0] if current else {}
        final_row = final[0] if final else {}
        result.append({
            "task": task,
            "acquisition_open": float(acquisition.get("open_accuracy", "nan")),
            "acquisition_restricted": float(acquisition.get("restricted_accuracy", "nan")),
            "final_open": float(final_row.get("open_accuracy", "nan")),
            "final_restricted": float(final_row.get("restricted_accuracy", "nan")),
        })
    old = [r for r in result if r["task"] < 5 and math.isfinite(r["final_open"])]
    bwt = float(np.mean([r["final_open"] - r["acquisition_open"] for r in old])) if old else float("nan")
    forgetting = float(np.mean([r["acquisition_open"] - r["final_open"] for r in old])) if old else float("nan")
    return result, bwt, forgetting


def write_report(run_dir, source_hash, candidates, selected, test_results, trajectory, bwt, forgetting, runtime_meta):
    original = test_results["original"]
    selected_test = test_results[selected["method"]]
    raw_delta = 100.0 * (original["open"] - 0.7076)
    cal_delta = 100.0 * (selected_test["open"] - 0.7076)
    val_gain = 100.0 * (selected["val_open"] - candidates["original"]["val_open"])
    test_gain = 100.0 * (selected_test["open"] - original["open"])
    lines = [
        "# CIFAR-100 job4971615 normalized Combined full report", "",
        f"Canonical source SHA256 before: `{source_hash}`",
        f"Canonical source SHA256 after: `{runtime_meta.get('canonical_source_sha256_after', source_hash)}`", "",
        "## Required comparison", "",
        "| Variant | All-seen | Restricted | Δ vs canonical |", "| --- | ---: | ---: | ---: |",
        f"| Canonical Combined | 70.76% | 94.78% | 0.00 pp |",
        f"| Normalized Combined | {100*original['open']:.3f}% | {100*original['restricted']:.3f}% | {raw_delta:+.3f} pp |",
        f"| Normalized Combined + Calibration | {100*selected_test['open']:.3f}% | {100*selected_test['restricted']:.3f}% | {cal_delta:+.3f} pp |",
        "", "## Calibration", "",
        f"Selected method: **{selected['method']}**",
        f"Validation selection gain: {val_gain:+.3f} pp open; test gain after calibration: {test_gain:+.3f} pp open.",
        f"Validation restricted guardrail: candidate={100*selected['val_restricted']:.3f}%, original={100*candidates['original']['val_restricted']:.3f}%.",
        "Fitting used validation logits only; test labels were used only for the final frozen evaluation.", "",
        "| Candidate | Val open | Val restricted | Test open | Test restricted |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("original", "scale", "bias", "affine"):
        c = candidates[name]
        t = test_results[name]
        lines.append(f"| {name} | {100*c['val_open']:.3f}% | {100*c['val_restricted']:.3f}% | {100*t['open']:.3f}% | {100*t['restricted']:.3f}% |")
    lines += ["", "## Acquisition, retention, BWT, forgetting", "", "| Task | Acquisition open | Acquisition restricted | Final open | Final restricted |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in trajectory:
        lines.append(f"| T{row['task']} | {row['acquisition_open']:.3f}% | {row['acquisition_restricted']:.3f}% | {row['final_open']:.3f}% | {row['final_restricted']:.3f}% |")
    lines += ["", f"BWT (open, mean T1–T4): {100*bwt:+.3f} pp", f"Forgetting (open, mean T1–T4): {100*forgetting:+.3f} pp", "", "## FactorOrth contribution trajectory", "", "See `tables/normalized_factororth_trajectory.csv` for T2–T5 per-epoch raw/normalized energy, weighted normalized energy, FactorOrth/CE, and FactorOrth/non-orthogonal-loss ratios.", "", "## Interpretation", "", "The raw normalized-Combined result isolates the training-side normalization effect. The calibrated result adds only validation-fitted task-block post-processing; it must not be described as raw training performance.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    val = np.load(run_dir / "validation_logits.npz")
    test = np.load(run_dir / "test_logits.npz")
    val_logits, val_labels = val["logits"].astype(np.float64), val["labels"].astype(np.int64)
    test_logits, test_labels = test["logits"].astype(np.float64), test["labels"].astype(np.int64)
    candidates = {
        "original": evaluate_candidate("original", val_logits, val_labels, []),
        "scale": evaluate_candidate("scale", val_logits, val_labels, fit_scale(val_logits, val_labels)),
        "bias": evaluate_candidate("bias", val_logits, val_labels, fit_bias(val_logits, val_labels)),
        "affine": evaluate_candidate("affine", val_logits, val_labels, fit_affine(val_logits, val_labels)),
    }
    original_restricted = candidates["original"]["val_restricted"]
    eligible = [c for c in candidates.values() if c["val_restricted"] >= original_restricted - 0.02]
    priority = {"original": 0, "scale": 1, "bias": 2, "affine": 3}
    selected = max(eligible, key=lambda c: (c["val_open"], -priority[c["method"]]))
    test_results = {}
    for name, candidate in candidates.items():
        logits = test_logits if name == "original" else apply_params(test_logits, name, candidate["params"])
        test_results[name] = {"open": open_accuracy(logits, test_labels), "restricted": restricted_accuracy(logits, test_labels)}
    (run_dir / "calibration_parameters.json").write_text(json.dumps({k: v["params"] for k, v in candidates.items()}, indent=2), encoding="utf-8")
    with (run_dir / "calibration_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "val_open", "val_restricted", "val_nll", "test_open", "test_restricted", "eligible", "selected"])
        writer.writeheader()
        for name, candidate in candidates.items():
            writer.writerow({"method": name, "val_open": candidate["val_open"], "val_restricted": candidate["val_restricted"], "val_nll": candidate["val_nll"], "test_open": test_results[name]["open"], "test_restricted": test_results[name]["restricted"], "eligible": candidate in eligible, "selected": name == selected["method"]})
    trajectory, bwt, forgetting = trajectory_summary(load_trajectory(run_dir))
    write_report(run_dir, metadata["canonical_source_sha256_before"], candidates, selected, test_results, trajectory, bwt, forgetting, metadata)
    print(f"SELECTED_CALIBRATION={selected['method']}")
    print(f"REPORT={REPORT_PATH}")
    print(f"RAW_OPEN={100*test_results['original']['open']:.3f}%")
    print(f"CALIBRATED_OPEN={100*test_results[selected['method']]['open']:.3f}%")


if __name__ == "__main__":
    main()
