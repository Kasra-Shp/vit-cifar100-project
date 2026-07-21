"""
Dry-run of calibrate_classifier_row_norms_confidence_weighted() against the
Fix 1 run's saved data (no training, no model weights available -- only the
pre_calibration mean row norms per (method, step) already logged in
tables/classifier_row_norm_diagnostics_by_method_step.csv, and the per-step
final-epoch val_ce_loss already logged in
tables/training_loss_history_by_epoch.csv).

This reproduces the NEW function's exact arithmetic (grouping, group target
norm = mean of per-step pre-cal norms in the group, boost = clamp(
(step_val_ce / group_mean_val_ce) ** gamma, boost_min, boost_max), target =
group_target_norm * boost) on those saved numbers, standing in for the actual
classifier weight tensors (which the run did not persist to disk -- same
constraint as analysis_recency_fix/report.txt Task B.5 for Fix 1's own
dry run).

Output: dry_run_row_norm_projection.csv (this directory).
"""
import csv
import math

FIX1_DIR = r"R3/results_fix1_20260721_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260721_172311"
ROW_NORM_CSV = FIX1_DIR + "/tables/classifier_row_norm_diagnostics_by_method_step.csv"
LOSS_CSV = FIX1_DIR + "/tables/training_loss_history_by_epoch.csv"

NUM_STEPS = 5
GAMMA = 0.5
BOOST_MIN = 0.85
BOOST_MAX = 1.30

RANKEXT_METHODS = {
    "rank_extension": False,
    "rank_extension_kd_only_T2": True,
    "rank_extension_orth_factor_lam_50": False,
    "rank_extension_orth_factor_lam_50_kd_T2": True,
}

# --- load pre-calibration row norms (method, step_id) -> mean_row_norm ---
pre_cal_norm = {}
with open(ROW_NORM_CSV) as f:
    for row in csv.DictReader(f):
        if row["phase"] != "pre_calibration":
            continue
        if row["method"] not in RANKEXT_METHODS:
            continue
        pre_cal_norm[(row["method"], int(row["step_id"]))] = float(row["mean_row_norm"])

# --- load final-epoch val_ce_loss per (method, step_id) ---
best_epoch = {}
final_val_ce = {}
with open(LOSS_CSV) as f:
    for row in csv.DictReader(f):
        m = row["method"]
        if m not in RANKEXT_METHODS:
            continue
        step_id = int(row["cl_step"])
        epoch = int(float(row["local_epoch"]))
        val_ce = row["val_ce_loss"]
        if val_ce in ("", "nan"):
            continue
        val_ce = float(val_ce)
        key = (m, step_id)
        if key not in best_epoch or epoch >= best_epoch[key]:
            best_epoch[key] = epoch
            final_val_ce[key] = val_ce

out_rows = []
for method, uses_kd in RANKEXT_METHODS.items():
    if uses_kd:
        groups = [[1], list(range(2, NUM_STEPS + 1))]
    else:
        groups = [list(range(1, NUM_STEPS + 1))]

    for group in groups:
        group_norms = [pre_cal_norm[(method, s)] for s in group]
        group_target_norm = sum(group_norms) / len(group_norms)

        group_val_ces = [final_val_ce[(method, s)] for s in group if (method, s) in final_val_ce]
        group_mean_val_ce = (sum(group_val_ces) / len(group_val_ces)) if group_val_ces else None

        for step_id in group:
            step_norm = pre_cal_norm[(method, step_id)]
            step_val_ce = final_val_ce.get((method, step_id))

            if uses_kd and len(group) > 1 and step_val_ce is not None and group_mean_val_ce and group_mean_val_ce > 1e-8:
                relative_difficulty = step_val_ce / group_mean_val_ce
                boost = min(max(relative_difficulty ** GAMMA, BOOST_MIN), BOOST_MAX)
            else:
                relative_difficulty = 1.0
                boost = 1.0

            target_norm = group_target_norm * boost
            equalization_ratio = target_norm / step_norm

            out_rows.append({
                "method": method,
                "step_id": step_id,
                "uses_kd": uses_kd,
                "group": "+".join(str(s) for s in group),
                "pre_cal_row_norm": round(step_norm, 6),
                "val_ce_loss_final_epoch": round(step_val_ce, 6) if step_val_ce is not None else "",
                "group_mean_val_ce_loss": round(group_mean_val_ce, 6) if group_mean_val_ce is not None else "",
                "relative_difficulty": round(relative_difficulty, 6),
                "boost_factor": round(boost, 6),
                "fix1_plain_group_mean_target_norm": round(group_target_norm, 6),
                "fix2_confidence_weighted_target_norm": round(target_norm, 6),
                "fix2_equalization_ratio_vs_pre_cal": round(equalization_ratio, 6),
            })

out_path = "analysis_recency_fix2/dry_run_row_norm_projection.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

print(f"Wrote {len(out_rows)} rows to {out_path}")
for r in out_rows:
    print(r)
