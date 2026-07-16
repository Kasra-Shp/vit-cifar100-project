"""
Comparative analysis: BASELINE (results_EPOCH6_20260711_light) vs NEW (results_calibfix_20260716_light)
Focus: rank_extension family first_step accuracy drops + factor_orth weakness.
Works only from saved CSVs -- no training.
"""
import pandas as pd
import numpy as np
import json

B_DIR = "R3/results_EPOCH6_20260711_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260711_214100"
N_DIR = "R3/results_calibfix_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_133338"
OUT = "analysis_rankext_firststep/"

DISPLAY = {
    "simple_avg": "SimpleAvg",
    "simple_avg_kd_T2": "SimpleAvg+KD",
    "simple_avg_factor_orth": "SimpleAvg+FactorOrth",
    "simple_avg_factor_orth_kd_T2": "SimpleAvg+FactorOrth+KD",
    "rank_extension": "RankExt",
    "rank_extension_kd_only_T2": "RankExt+KD",
    "rank_extension_orth_factor_lam_50": "RankExt+FactorOrth",
    "rank_extension_orth_factor_lam_50_kd_T2": "RankExt+FactorOrth+KD",
}
RANKEXT_METHODS = ["rank_extension", "rank_extension_kd_only_T2",
                    "rank_extension_orth_factor_lam_50", "rank_extension_orth_factor_lam_50_kd_T2"]

# ============================================================
# Load core tables
# ============================================================
b_acc = pd.read_csv(f"{B_DIR}/tables/all_results_selected_methods.csv")
n_acc = pd.read_csv(f"{N_DIR}/tables/all_results_selected_methods.csv")
b_final = pd.read_csv(f"{B_DIR}/tables/final_metrics_all_methods.csv")
n_final = pd.read_csv(f"{N_DIR}/tables/final_metrics_all_methods.csv")
b_hist = pd.read_csv(f"{B_DIR}/tables/training_loss_history_by_epoch.csv")
n_hist = pd.read_csv(f"{N_DIR}/tables/training_loss_history_by_epoch.csv")
n_best_epoch = pd.read_csv(f"{N_DIR}/tables/best_epoch_selected_by_method_step.csv")

# per-method config confirmation (calibration / head-lr)
n_hparams = json.load(open(f"{N_DIR}/configs/hyperparameters_by_method.json"))
b_hparams = json.load(open(f"{B_DIR}/configs/hyperparameters_by_method.json"))

# ============================================================
# TABLE 1: accuracy/loss comparison, all 8 methods x 3 eval sets
# ============================================================
merged = b_acc.merge(n_acc, on=["method", "eval_set"], suffixes=("_BASELINE", "_NEW"))
merged["delta_accuracy_pp"] = (merged.accuracy_NEW - merged.accuracy_BASELINE) * 100
merged["delta_loss"] = merged.loss_NEW - merged.loss_BASELINE
merged["accuracy_BASELINE_pct"] = merged.accuracy_BASELINE * 100
merged["accuracy_NEW_pct"] = merged.accuracy_NEW * 100
merged["family"] = merged.method.apply(lambda m: "rank_extension" if m.startswith("rank_extension") else "simple_avg")
merged["uses_kd"] = merged.method.str.contains("kd")
merged["uses_orth"] = merged.method.str.contains("orth")
merged["display_name"] = merged.method.map(DISPLAY)
cols = ["method", "display_name", "family", "uses_orth", "uses_kd", "eval_set",
        "accuracy_BASELINE_pct", "accuracy_NEW_pct", "delta_accuracy_pp",
        "loss_BASELINE", "loss_NEW", "delta_loss"]
table1 = merged[cols].sort_values(["family", "uses_orth", "uses_kd", "eval_set"])
table1.to_csv(f"{OUT}table1_accuracy_comparison_all_methods.csv", index=False)

# ============================================================
# TABLE 2: rank_ext per-step accuracy (NEW only, per-step list available)
#          + forgetting_metric / backward_transfer both runs
# ============================================================
rows = []
for m in RANKEXT_METHODS:
    brow = b_final[b_final.method == m].iloc[0]
    nrow = n_final[n_final.method == m].iloc[0]
    per_step_new = eval(nrow.per_step_accuracy) if isinstance(nrow.per_step_accuracy, str) else nrow.per_step_accuracy
    rows.append({
        "method": m, "display_name": DISPLAY[m],
        "BASELINE_first_step_acc": brow.first_step_accuracy,
        "NEW_first_step_acc": nrow.first_step_accuracy,
        "delta_first_step_pp": nrow.first_step_accuracy - brow.first_step_accuracy,
        "BASELINE_later_steps_acc": brow.later_steps_accuracy,
        "NEW_later_steps_acc": nrow.later_steps_accuracy,
        "delta_later_steps_pp": nrow.later_steps_accuracy - brow.later_steps_accuracy,
        "BASELINE_all_seen_acc": brow.all_seen_accuracy,
        "NEW_all_seen_acc": nrow.all_seen_accuracy,
        "delta_all_seen_pp": nrow.all_seen_accuracy - brow.all_seen_accuracy,
        "NEW_per_step_acc_1to5": per_step_new,
        "BASELINE_forgetting_metric": brow.forgetting_metric,
        "NEW_forgetting_metric": nrow.forgetting_metric,
        "NEW_backward_transfer": nrow.backward_transfer,
        "NEW_forward_transfer": nrow.forward_transfer,
    })
table2 = pd.DataFrame(rows)
table2.to_csv(f"{OUT}table2_rankext_perstep_and_forgetting.csv", index=False)

# ============================================================
# TABLE 3: step-1 training-time val CE vs step-1 final (post all-steps) eval loss/acc
#   -- distinguishes training-time cause vs forgetting-time cause
# ============================================================
rows = []
for m in RANKEXT_METHODS:
    # BASELINE: final epoch of cl_step==1 (no best-epoch selection existed)
    b_step1 = b_hist[(b_hist.method == m) & (b_hist.cl_step == 1)].sort_values("local_epoch")
    b_step1_end_val_ce = b_step1.iloc[-1].val_ce_loss
    b_step1_best_val_ce = b_step1.val_ce_loss.min()

    # NEW: both final-epoch and best-epoch-selected value (best-epoch selection active)
    n_step1 = n_hist[(n_hist.method == m) & (n_hist.cl_step == 1)].sort_values("local_epoch")
    n_step1_end_val_ce = n_step1.iloc[-1].val_ce_loss
    n_best_row = n_best_epoch[(n_best_epoch.method_name == m) & (n_best_epoch.step_id == 1)].iloc[0]
    n_step1_selected_val_ce = n_best_row.selected_val_ce

    b_eval = b_acc[(b_acc.method == m) & (b_acc.eval_set == "first_step")].iloc[0]
    n_eval = n_acc[(n_acc.method == m) & (n_acc.eval_set == "first_step")].iloc[0]

    rows.append({
        "method": m, "display_name": DISPLAY[m],
        "BASELINE_step1_training_end_val_ce": b_step1_end_val_ce,
        "BASELINE_step1_training_best_val_ce": b_step1_best_val_ce,
        "BASELINE_step1_FINAL_eval_loss_post_all_steps": b_eval.loss,
        "BASELINE_step1_FINAL_eval_acc_post_all_steps": b_eval.accuracy * 100,
        "BASELINE_loss_inflation_factor": b_eval.loss / b_step1_end_val_ce,
        "NEW_step1_training_end_val_ce_ep9": n_step1_end_val_ce,
        "NEW_step1_training_selected_val_ce_best_epoch": n_step1_selected_val_ce,
        "NEW_step1_FINAL_eval_loss_post_all_steps": n_eval.loss,
        "NEW_step1_FINAL_eval_acc_post_all_steps": n_eval.accuracy * 100,
        "NEW_loss_inflation_factor": n_eval.loss / n_step1_selected_val_ce,
    })
table3 = pd.DataFrame(rows)
table3.to_csv(f"{OUT}table3_step1_trainingtime_vs_evaltime.csv", index=False)

# ============================================================
# TABLE 4: factor-orth trajectory stats: rank_ext+orth vs simple_avg+orth (both KD states, both runs)
#   step-boundary (local_epoch==1) train_ce + factor_orth_weighted, compared to the no-orth sibling
#   at the same step boundary, per run.
# ============================================================
PAIRS = [
    ("rank_extension_orth_factor_lam_50", "rank_extension"),
    ("rank_extension_orth_factor_lam_50_kd_T2", "rank_extension_kd_only_T2"),
    ("simple_avg_factor_orth", "simple_avg"),
    ("simple_avg_factor_orth_kd_T2", "simple_avg_kd_T2"),
]
rows = []
for label, hist in [("BASELINE", b_hist), ("NEW", n_hist)]:
    for orth_m, plain_m in PAIRS:
        orth_ep1 = hist[(hist.method == orth_m) & (hist.local_epoch == 1) & (hist.cl_step > 1)]
        plain_ep1 = hist[(hist.method == plain_m) & (hist.local_epoch == 1) & (hist.cl_step > 1)]
        merged_pair = orth_ep1.merge(plain_ep1, on="cl_step", suffixes=("_orth", "_plain"))
        for _, r in merged_pair.iterrows():
            rows.append({
                "run": label, "orth_method": orth_m, "plain_method": plain_m,
                "cl_step": r.cl_step,
                "orth_train_ce_at_step_start": r.train_ce_loss_orth,
                "plain_train_ce_at_step_start": r.train_ce_loss_plain,
                "ce_ratio_orth_over_plain": r.train_ce_loss_orth / r.train_ce_loss_plain,
                "factor_orth_loss_raw_at_step_start": r.factor_orth_loss_raw_orth,
                "factor_orth_loss_weighted_at_step_start": r.factor_orth_loss_weighted_orth,
                "factor_orth_weighted_over_ce": r.factor_orth_loss_weighted_orth / r.train_ce_loss_orth,
            })
table4 = pd.DataFrame(rows)
table4.to_csv(f"{OUT}table4_factororth_trajectory_stats.csv", index=False)

# also a compact end-of-training-step summary (last epoch per cl_step) to show reconvergence
rows = []
for label, hist in [("BASELINE", b_hist), ("NEW", n_hist)]:
    for orth_m, plain_m in PAIRS:
        for cl_step in sorted(hist.cl_step.unique()):
            o = hist[(hist.method == orth_m) & (hist.cl_step == cl_step)].sort_values("local_epoch")
            p = hist[(hist.method == plain_m) & (hist.cl_step == cl_step)].sort_values("local_epoch")
            if len(o) == 0 or len(p) == 0:
                continue
            rows.append({
                "run": label, "orth_method": orth_m, "plain_method": plain_m, "cl_step": cl_step,
                "orth_final_epoch_val_ce": o.iloc[-1].val_ce_loss,
                "plain_final_epoch_val_ce": p.iloc[-1].val_ce_loss,
                "val_ce_diff_orth_minus_plain": o.iloc[-1].val_ce_loss - p.iloc[-1].val_ce_loss,
            })
table4b = pd.DataFrame(rows)
table4b.to_csv(f"{OUT}table4b_factororth_endofstep_reconvergence.csv", index=False)

# ============================================================
# TABLE 5: rank_ext vs rank_ext+orth head-to-head (plain vs orth, KD and non-KD, both runs)
# ============================================================
rows = []
HEAD_TO_HEAD = [
    ("rank_extension", "rank_extension_orth_factor_lam_50", "non-KD"),
    ("rank_extension_kd_only_T2", "rank_extension_orth_factor_lam_50_kd_T2", "KD"),
]
for label, acc_df in [("BASELINE", b_acc), ("NEW", n_acc)]:
    for plain_m, orth_m, kd_state in HEAD_TO_HEAD:
        for eval_set in ["first_step", "later_steps", "all_seen"]:
            p = acc_df[(acc_df.method == plain_m) & (acc_df.eval_set == eval_set)].iloc[0]
            o = acc_df[(acc_df.method == orth_m) & (acc_df.eval_set == eval_set)].iloc[0]
            rows.append({
                "run": label, "kd_state": kd_state, "eval_set": eval_set,
                "plain_method": plain_m, "orth_method": orth_m,
                "plain_acc_pct": p.accuracy * 100, "orth_acc_pct": o.accuracy * 100,
                "orth_minus_plain_pp": (o.accuracy - p.accuracy) * 100,
                "plain_loss": p.loss, "orth_loss": o.loss, "loss_delta": o.loss - p.loss,
            })
table5 = pd.DataFrame(rows)
table5.to_csv(f"{OUT}table5_rankext_vs_rankext_orth_headtohead.csv", index=False)

# ============================================================
# TABLE 6: rank_ext+orth vs simple_avg+orth "counterpart gap" (as framed in the task)
# ============================================================
rows = []
COUNTERPART_PAIRS = [
    ("rank_extension_orth_factor_lam_50", "simple_avg_factor_orth", "non-KD"),
    ("rank_extension_orth_factor_lam_50_kd_T2", "simple_avg_factor_orth_kd_T2", "KD"),
]
for label, acc_df in [("BASELINE", b_acc), ("NEW", n_acc)]:
    for rext_m, savg_m, kd_state in COUNTERPART_PAIRS:
        for eval_set in ["first_step", "later_steps", "all_seen"]:
            r = acc_df[(acc_df.method == rext_m) & (acc_df.eval_set == eval_set)].iloc[0]
            s = acc_df[(acc_df.method == savg_m) & (acc_df.eval_set == eval_set)].iloc[0]
            rows.append({
                "run": label, "kd_state": kd_state, "eval_set": eval_set,
                "rankext_orth_acc_pct": r.accuracy * 100, "simpleavg_orth_acc_pct": s.accuracy * 100,
                "rankext_minus_simpleavg_pp": (r.accuracy - s.accuracy) * 100,
            })
table6 = pd.DataFrame(rows)
table6.to_csv(f"{OUT}table6_rankext_orth_vs_simpleavg_orth_counterpart_gap.csv", index=False)

# ============================================================
# TABLE 7: hyperparameter confirmation (calibration / head-lr) per method, both runs
# ============================================================
rows = []
for item in n_hparams:
    rows.append({"run": "NEW", "display_name": item.get("display_method_name"),
                 "apply_calibration": item.get("apply_calibration"),
                 "head_lr_multiplier": item.get("head_lr_multiplier"),
                 "lambda_factor_orth": item.get("lambda_factor_orth")})
for item in b_hparams:
    rows.append({"run": "BASELINE", "display_name": item.get("display_method_name"),
                 "apply_calibration": item.get("apply_calibration"),
                 "head_lr_multiplier": item.get("head_lr_multiplier"),
                 "lambda_factor_orth": item.get("lambda_factor_orth")})
table7 = pd.DataFrame(rows)
table7.to_csv(f"{OUT}table7_hyperparameter_confirmation.csv", index=False)

print("=== TABLE 1 (rank_ext rows only) ===")
print(table1[table1.family == "rank_extension"].to_string(index=False))
print("\n=== TABLE 2 ===")
print(table2.to_string(index=False))
print("\n=== TABLE 3 ===")
print(table3.to_string(index=False))
print("\n=== TABLE 5 ===")
print(table5.to_string(index=False))
print("\n=== TABLE 6 ===")
print(table6.to_string(index=False))
print("\nAll tables written to", OUT)
