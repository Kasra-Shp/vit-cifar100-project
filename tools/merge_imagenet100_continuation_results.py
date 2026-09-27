"""Merge the audited 5 completed methods with continuation methods 6-8.

The command is intentionally conservative: it discovers existing summary
tables, requires exact method coverage, compares protocol manifests, and
never creates a row for a missing method.  A completed methods-6/7
continuation and a separately resumed Method-8 result may be supplied as two
inputs; the legacy single-result form remains supported for a complete 6-8
continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


COMPLETED_METHODS = {
    "simple_avg",
    "simple_avg_kd_oldseen_T2_warmup",
    "simple_avg_dense_orth_lam20",
    "simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup",
    "rank_extension",
}
CONTINUATION_METHODS = {
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_lam50",
    "rank_extension_factor_orth_lam50_fullkd_T2_protect30",
}
CONTINUATION_METHODS_6_7 = {
    "rank_extension_fullkd_T2_protect30",
    "rank_extension_factor_orth_lam50",
}
RESUMED_METHOD_8 = {"rank_extension_factor_orth_lam50_fullkd_T2_protect30"}
PROTOCOL_KEYS = {
    "dataset_identifier", "wnid_split_sha256", "seed", "num_classes",
    "num_tasks", "classes_per_task", "epochs", "preprocessing",
    "rank_schedule", "lr_rankext", "head_lr_multiplier_rankext",
    "kd_temperature", "kd_weight", "protect_weight", "factor_orth_lambda",
    "classifier_restoration", "evaluation",
}


def _find_summary(root: Path) -> Path:
    candidates = sorted((root / "tables").glob("*summary_table.csv"))
    if not candidates:
        raise FileNotFoundError(f"no *summary_table.csv found under {root / 'tables'}")
    if len(candidates) > 1:
        preferred = [p for p in candidates if p.name in {
            "final_8method_summary_table.csv",
            "final_9method_summary_table.csv",
        }]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(f"ambiguous summary tables under {root}: {candidates}")
    return candidates[0]


def _manifest_path(result_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        result_root / "configs" / "protocol_manifest.json",
        result_root / "protocol_manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    canonical = Path("experiments_prepared/imagenet100_5x20_canonical_protocol_manifest.json")
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(
        f"protocol manifest missing for {result_root} and canonical manifest is unavailable"
    )


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_identifier", "wnid_split_sha256", "seed", "num_classes",
        "num_tasks", "classes_per_task", "epochs", "preprocessing",
        "rank_schedule", "lr_rankext", "head_lr_multiplier_rankext",
        "kd_temperature", "kd_weight", "protect_weight", "factor_orth_lambda",
        "classifier_restoration", "evaluation",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"manifest {path} is missing required fields: {missing}")
    return payload


def _verify_protocol(original: dict, continuation: dict) -> None:
    if any(original.get(key) != continuation.get(key) for key in PROTOCOL_KEYS):
        differences = {
            key: (original.get(key), continuation.get(key))
            for key in sorted(PROTOCOL_KEYS)
            if original.get(key) != continuation.get(key)
        }
        raise ValueError(f"protocol mismatch; refusing merge: {differences}")
    if original["dataset_identifier"] != "ImageNet-100/PODNet-DER-DyTox-first100-sorted-wnids":
        raise ValueError("unexpected dataset identity")
    if original["wnid_split_sha256"] != hashlib.sha256(
        Path("experiments_prepared/splits/imagenet100_class_order_seed42.json").read_bytes()
    ).hexdigest():
        raise ValueError("WNID/task split hash does not match the checked-in seed42 artifact")
    if original["seed"] != 42 or original["num_tasks"] != 5 or original["classes_per_task"] != 20:
        raise ValueError("seed/task protocol is not the canonical 5x20 seed42 protocol")
    if original["epochs"] != 9 or original["rank_schedule"] != [16, 32, 48, 64, 80]:
        raise ValueError("epoch or RankExt schedule mismatch")


def merge(
    original_root: Path,
    continuation_root: Path,
    output: Path,
    original_manifest: Path | None,
    resumed_method8_root: Path | None = None,
) -> None:
    original_summary = pd.read_csv(_find_summary(original_root))
    continuation_summary = pd.read_csv(_find_summary(continuation_root))
    if "method" not in original_summary or "method" not in continuation_summary:
        raise ValueError("summary tables must contain a method column")
    original_names = set(original_summary["method"].dropna())
    continuation_names = set(continuation_summary["method"].dropna())
    if original_names != COMPLETED_METHODS:
        raise ValueError(f"original table coverage mismatch: {sorted(original_names)}")
    if resumed_method8_root is None:
        if continuation_names != CONTINUATION_METHODS:
            raise ValueError(f"continuation table coverage mismatch: {sorted(continuation_names)}")
        continuation_parts = [continuation_summary]
    else:
        resumed_summary = pd.read_csv(_find_summary(resumed_method8_root))
        if "method" not in resumed_summary:
            raise ValueError("resumed Method-8 summary must contain a method column")
        resumed_names = set(resumed_summary["method"].dropna())
        if continuation_names != CONTINUATION_METHODS_6_7:
            raise ValueError(
                "methods-6/7 continuation coverage mismatch: "
                f"{sorted(continuation_names)}"
            )
        if resumed_names != RESUMED_METHOD_8:
            raise ValueError(f"resumed Method-8 coverage mismatch: {sorted(resumed_names)}")
        continuation_parts = [continuation_summary, resumed_summary]
        resumed_protocol = _load_manifest(_manifest_path(resumed_method8_root, None))
        continuation_protocol = _load_manifest(_manifest_path(continuation_root, None))
        _verify_protocol(continuation_protocol, resumed_protocol)
    all_continuation_names = set().union(*(set(part["method"].dropna()) for part in continuation_parts))
    if original_names & all_continuation_names:
        raise ValueError("duplicate method rows would be overwritten")

    original_protocol = _load_manifest(_manifest_path(original_root, original_manifest))
    continuation_protocol = _load_manifest(_manifest_path(continuation_root, None))
    _verify_protocol(original_protocol, continuation_protocol)

    if any(list(original_summary.columns) != list(part.columns) for part in continuation_parts):
        raise ValueError("summary schemas differ; refusing to silently reshape metrics")
    merged = pd.concat([original_summary, *continuation_parts], ignore_index=True)
    if len(merged) != 8 or set(merged["method"]) != COMPLETED_METHODS | CONTINUATION_METHODS:
        raise AssertionError("merged table does not contain exactly the audited 8 methods")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"MERGE PASS: wrote {len(merged)} audited rows -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--continuation-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-protocol-manifest", type=Path)
    parser.add_argument(
        "--resumed-method8-result", type=Path,
        help="separate final Method-8 result root; continuation-result must then contain only methods 6-7",
    )
    args = parser.parse_args()
    merge(
        args.original_result, args.continuation_result, args.output,
        args.original_protocol_manifest, args.resumed_method8_result,
    )


if __name__ == "__main__":
    main()
