"""
Rigorous OLD (EPOCH3, 10 methods, dropout 0.05) vs NEW (EPOCH6, 8 methods,
dropout 0.1, best-epoch selection) comparison.

Reads ONLY the saved CSVs under R3/ -- no retraining, no live model access.
Reproduces every table/figure under analysis_R4/{tables,plots,reports}.

Run with:  python analysis_R4/generate_analysis_R4.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
OLD_BASE = os.path.join(
    "R3", "results_MAIN_20260709_light",
    "clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260709_083052",
)
NEW_BASE = os.path.join(
    "R3", "results_EPOCH6_20260711_light",
    "clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260711_214100",
)
OLD_TABLES = os.path.join(OLD_BASE, "tables")
NEW_TABLES = os.path.join(NEW_BASE, "tables")

OUT_BASE = "analysis_R4"
OUT_TABLES = os.path.join(OUT_BASE, "tables")
OUT_PLOTS = os.path.join(OUT_BASE, "plots")
OUT_REPORTS = os.path.join(OUT_BASE, "reports")
for d in (OUT_TABLES, OUT_PLOTS, OUT_REPORTS):
    os.makedirs(d, exist_ok=True)

print("OLD run resolved path:", os.path.abspath(OLD_BASE))
print("NEW run resolved path:", os.path.abspath(NEW_BASE))

# ----------------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})

SHARED_METHODS = [
    "simple_avg",
    "simple_avg_kd_T2",
    "simple_avg_factor_orth",
    "simple_avg_factor_orth_kd_T2",
    "rank_extension",
    "rank_extension_kd_only_T2",
    "rank_extension_orth_factor_lam_50",
    "rank_extension_orth_factor_lam_50_kd_T2",
]
DISPLAY_NAME = {
    "simple_avg": "SimpleAvg",
    "simple_avg_kd_T2": "SimpleAvg+KD",
    "simple_avg_factor_orth": "SimpleAvg+FactorOrth",
    "simple_avg_factor_orth_kd_T2": "SimpleAvg+FactorOrth+KD",
    "rank_extension": "RankExt",
    "rank_extension_kd_only_T2": "RankExt+KD",
    "rank_extension_orth_factor_lam_50": "RankExt+FactorOrth",
    "rank_extension_orth_factor_lam_50_kd_T2": "RankExt+FactorOrth+KD",
    "simple_avg_delta_orth": "SimpleAvg+DeltaTrace (OLD only)",
    "rank_extension_orth_delta_trace_lam_50": "RankExt+DeltaTrace (OLD only)",
}
TAB10 = plt.get_cmap("tab10").colors
METHOD_COLOR = {m: TAB10[i % 10] for i, m in enumerate(SHARED_METHODS)}


def dname(m):
    return DISPLAY_NAME.get(m, m)


def outside_legend(ax, **kw):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kw)


# ----------------------------------------------------------------------------
# Load core data
# ----------------------------------------------------------------------------
old_hist = pd.read_csv(os.path.join(OLD_TABLES, "training_loss_history_by_epoch.csv"))
old_hist = old_hist[["method_name", "step_id", "epoch", "train_ce_loss", "val_ce_loss"]].dropna(
    subset=["train_ce_loss"]
).sort_values(["method_name", "step_id", "epoch"]).reset_index(drop=True)

new_hist = pd.read_csv(os.path.join(NEW_TABLES, "all_methods_convergence_table.csv"))
new_hist = new_hist[["method_name", "step_id", "epoch", "train_ce_loss", "val_ce_loss"]].sort_values(
    ["method_name", "step_id", "epoch"]
).reset_index(drop=True)

old_ranking = pd.read_csv(os.path.join(OLD_TABLES, "ranking_by_all_seen_selected_methods.csv"))
new_ranking = pd.read_csv(os.path.join(NEW_TABLES, "ranking_by_all_seen_selected_methods.csv"))

new_valdiag = pd.read_csv(os.path.join(NEW_TABLES, "validation_diagnostics_by_method.csv"))
new_hpcheck = pd.read_csv(os.path.join(NEW_TABLES, "hyperparameter_consistency_check.csv"))
new_finalmetrics = pd.read_csv(os.path.join(NEW_TABLES, "final_metrics_all_methods.csv"))
new_supervisor_acc = pd.read_csv(os.path.join(NEW_TABLES, "supervisor_selected_accuracy_comparison.csv"))

OLD_EPOCHS_PER_STEP = int(old_hist["epoch"].max())  # 3
NEW_EPOCHS_PER_STEP = int(new_hist["epoch"].max())  # 6
N_STEPS = 5
OLD_N_METHODS = old_hist["method_name"].nunique()  # 10
NEW_N_METHODS = new_hist["method_name"].nunique()  # 8

print(f"OLD: {OLD_N_METHODS} methods x {N_STEPS} steps x {OLD_EPOCHS_PER_STEP} epochs")
print(f"NEW: {NEW_N_METHODS} methods x {N_STEPS} steps x {NEW_EPOCHS_PER_STEP} epochs")

# ============================================================================
# Q5 -- SANITY & CONSISTENCY CHECKS (computed first; feeds report + used by others)
# ============================================================================
sanity_lines = []

expected_new_rows = NEW_N_METHODS * N_STEPS * NEW_EPOCHS_PER_STEP
actual_new_rows = len(new_hist)
sanity_lines.append(
    f"[ROW COUNT] all_methods_convergence_table.csv: expected {NEW_N_METHODS} methods x {N_STEPS} steps x "
    f"{NEW_EPOCHS_PER_STEP} epochs = {expected_new_rows} rows; actual = {actual_new_rows} rows -> "
    f"{'PASS' if expected_new_rows == actual_new_rows else 'FAIL'}"
)
# cross-check against the raw merged history-by-epoch table too
raw_new_hist = pd.read_csv(os.path.join(NEW_TABLES, "training_loss_history_by_epoch.csv"))
raw_rows_nonan = raw_new_hist.dropna(subset=["train_ce_loss"]).drop_duplicates(
    subset=["method_name", "step_id", "epoch"]
)
sanity_lines.append(
    f"[ROW COUNT cross-check] training_loss_history_by_epoch.csv (deduped method_name/step_id/epoch, "
    f"non-null train_ce_loss): {len(raw_rows_nonan)} unique rows -> "
    f"{'PASS' if len(raw_rows_nonan) == expected_new_rows else 'FAIL'}"
)

kd_temps = sorted(new_hpcheck["kd_temperature"].unique().tolist())
sanity_lines.append(
    f"[KD TEMPERATURE] distinct kd_temperature values across 8 methods: {kd_temps} -> "
    f"{'PASS (only 0.0 / 2.0)' if set(kd_temps).issubset({0.0, 2.0}) else 'FAIL'}"
)
lam_orth = sorted(new_hpcheck["lambda_orth"].unique().tolist())
sanity_lines.append(
    f"[LAMBDA_ORTH] distinct lambda_orth values across 8 methods: {lam_orth} -> "
    f"{'PASS (only 0.0 / 50.0)' if set(lam_orth).issubset({0.0, 50.0}) else 'FAIL'}"
)
delta_trace_methods = [m for m in new_hist["method_name"].unique() if "delta_trace" in m or "delta_orth" in m]
sanity_lines.append(
    f"[DELTA-TRACE ABSENCE] methods with delta-trace in name found in NEW run: {delta_trace_methods} -> "
    f"{'PASS (none present)' if len(delta_trace_methods) == 0 else 'FAIL'}"
)
methods_new = sorted(new_hist["method_name"].unique().tolist())
methods_expected = sorted(SHARED_METHODS)
sanity_lines.append(
    f"[METHOD SET] NEW run methods ({len(methods_new)}): {methods_new}\n"
    f"    matches expected 8-method set: {'PASS' if methods_new == methods_expected else 'FAIL'}"
)

live_conv_dir = os.path.join(NEW_BASE, "plots")
live_conv_pngs = [f for f in os.listdir(live_conv_dir) if f.startswith("live_convergence_")]
sanity_lines.append(
    f"[LIVE CONVERGENCE PLOTS] found {len(live_conv_pngs)} live_convergence_*.png files "
    f"(expected {NEW_N_METHODS}, one per method) -> "
    f"{'PASS' if len(live_conv_pngs) == NEW_N_METHODS else 'FAIL'}: {sorted(live_conv_pngs)}"
)

# per-step (1..5) accuracy trajectory check
per_step_acc_populated = new_supervisor_acc["per_step_accuracy"].notna().sum()
bwd_populated = new_supervisor_acc["backward_transfer"].notna().sum()
fwd_populated = new_supervisor_acc["forward_transfer"].notna().sum()
sanity_lines.append(
    f"[PER-STEP ACCURACY] 'per_step_accuracy' column EXISTS in supervisor_selected_accuracy_comparison.csv "
    f"(schema improvement over OLD run, which lacks the column entirely) but is POPULATED for "
    f"{per_step_acc_populated}/{len(new_supervisor_acc)} methods -> FAIL (schema added, data not delivered). "
    f"Same for backward_transfer ({bwd_populated}/8 populated) and forward_transfer ({fwd_populated}/8 populated). "
    f"Only first_step / later_steps / all_seen (3 aggregated eval groups) are populated, identical in kind "
    f"to the OLD run's export -- true 5-step-by-5-step accuracy is still not retained in either run."
)

sanity_lines.append(
    f"[best-epoch selection global-diagnostic] validation_diagnostics_by_method.csv reports a "
    f"'best_validation_ce' computed as the GLOBAL minimum val CE across all 30 concatenated "
    f"(cl_step, local_epoch) pairs per method, not a per-step local minimum. Because early CL "
    f"steps (fewer classes seen) have structurally lower val CE than later steps, this global "
    f"minimum is biased toward cl_step 1 and should NOT be read as 'the model was best at global "
    f"epoch X' in a comparable-difficulty sense. A separate per-step local best-epoch computation "
    f"is done below (Q3) for the meaningful within-step early-stopping question."
)

with open(os.path.join(OUT_REPORTS, "sanity_checks.txt"), "w") as f:
    f.write("SANITY & CONSISTENCY CHECKS -- NEW RUN\n" + "=" * 78 + "\n\n")
    f.write("\n\n".join(sanity_lines))
print("\n".join(sanity_lines))

# ============================================================================
# Q1 -- CONVERGENCE VERDICT (same test as OLD run's generate_analysis.py)
# ============================================================================
REL_IMPR_STILL_IMPROVING_THRESHOLD = 3.0
DIVERGE_THRESHOLD = -1.0


def rel_improvement(prev, curr):
    if prev is None or pd.isna(prev) or prev == 0:
        return np.nan
    return (prev - curr) / abs(prev) * 100.0


def verdict_from_rel(rel):
    if pd.isna(rel):
        return "n/a"
    if rel >= REL_IMPR_STILL_IMPROVING_THRESHOLD:
        return "still_improving"
    if rel <= DIVERGE_THRESHOLD:
        return "diverging"
    return "converged"


def build_convergence_table(hist_df, methods, epochs_per_step):
    rows = []
    for method in methods:
        for step in range(1, N_STEPS + 1):
            g = hist_df[(hist_df["method_name"] == method) & (hist_df["step_id"] == step)].sort_values("epoch")
            if g.empty:
                continue
            tce = g["train_ce_loss"].tolist()
            vce = g["val_ce_loss"].tolist()
            ep = g["epoch"].tolist()
            rel_t = rel_improvement(tce[-2], tce[-1]) if len(tce) >= 2 else np.nan
            rel_v = rel_improvement(vce[-2], vce[-1]) if len(vce) >= 2 else np.nan
            train_verdict = verdict_from_rel(rel_t)
            val_verdict = verdict_from_rel(rel_v)
            rows.append(dict(
                method_name=method, display_name=dname(method), step_id=step,
                final_epoch=ep[-1],
                final_epoch_rel_train_improvement_pct=round(rel_t, 3) if not pd.isna(rel_t) else np.nan,
                train_convergence_verdict=train_verdict,
                final_epoch_rel_val_improvement_pct=round(rel_v, 3) if not pd.isna(rel_v) else np.nan,
                val_convergence_verdict=val_verdict,
            ))
    return pd.DataFrame(rows)


new_verdict = build_convergence_table(new_hist, SHARED_METHODS, NEW_EPOCHS_PER_STEP)
old_verdict = build_convergence_table(old_hist, SHARED_METHODS, OLD_EPOCHS_PER_STEP)  # 8-method subset of OLD, for reference

new_verdict.to_csv(os.path.join(OUT_TABLES, "convergence_verdict_new_run.csv"), index=False)
old_verdict.to_csv(os.path.join(OUT_TABLES, "convergence_verdict_old_run_8method_subset.csv"), index=False)

n_total_new = len(new_verdict)
n_still_new = (new_verdict["train_convergence_verdict"] == "still_improving").sum()
n_converged_new = (new_verdict["train_convergence_verdict"] == "converged").sum()
n_diverging_new = (new_verdict["train_convergence_verdict"] == "diverging").sum()
mean_rel_new = new_verdict["final_epoch_rel_train_improvement_pct"].mean()

n_total_old = len(old_verdict)
n_still_old = (old_verdict["train_convergence_verdict"] == "still_improving").sum()

# val-side too
n_val_still_new = (new_verdict["val_convergence_verdict"] == "still_improving").sum()
n_val_converged_new = (new_verdict["val_convergence_verdict"] == "converged").sum()
n_val_diverging_new = (new_verdict["val_convergence_verdict"] == "diverging").sum()

# Per-method verdict classification: fully converged / marginally converged / still under-trained
per_method_class = []
for method in SHARED_METHODS:
    sub = new_verdict[new_verdict["method_name"] == method]
    n_si = (sub["train_convergence_verdict"] == "still_improving").sum()
    mean_rel = sub["final_epoch_rel_train_improvement_pct"].mean()
    if n_si == 0:
        cls = "fully converged"
    elif n_si <= 2:
        cls = "marginally converged"
    else:
        cls = "still under-trained"
    per_method_class.append(dict(
        method_name=method, display_name=dname(method),
        steps_still_improving_train=int(n_si), steps_total=5,
        mean_final_epoch_rel_train_improvement_pct=round(mean_rel, 2),
        classification=cls,
    ))
per_method_class_df = pd.DataFrame(per_method_class).sort_values(
    "mean_final_epoch_rel_train_improvement_pct", ascending=False
)
per_method_class_df.to_csv(os.path.join(OUT_TABLES, "per_method_convergence_classification.csv"), index=False)

# Epoch-extrapolation: use the LAST TWO transitions available (epoch4->5, epoch5->6) as the
# decay-rate seed, since that is what determines how many further epochs the CURRENT tail
# behaviour needs (the OLD script used epoch1->2 / epoch2->3 simply because that's all a
# 3-epoch curve has -- those ARE its last two transitions. For NEW's 6-epoch curve we use
# ITS last two transitions for a like-for-like "current decay rate" comparison).
CONVERGED_THRESHOLD_FRAC = REL_IMPR_STILL_IMPROVING_THRESHOLD / 100.0
extra_epochs_rows = []
for method in SHARED_METHODS:
    for step in range(1, N_STEPS + 1):
        g = new_hist[(new_hist["method_name"] == method) & (new_hist["step_id"] == step)].sort_values("epoch")
        tce = g["train_ce_loss"].tolist()
        if len(tce) < 3:
            continue
        r1 = rel_improvement(tce[-3], tce[-2]) / 100.0  # epoch4->5
        r2 = rel_improvement(tce[-2], tce[-1]) / 100.0  # epoch5->6 (== final-epoch verdict transition)
        if r1 <= 0 or r2 <= 0 or r2 >= r1:
            decay_ratio = 0.55
        else:
            decay_ratio = r2 / r1
        r = r2
        extra = 0
        while r >= CONVERGED_THRESHOLD_FRAC and extra < 30:
            r *= decay_ratio
            extra += 1
        extra_epochs_rows.append(dict(method_name=method, step_id=step, extra_epochs_needed=extra))
extra_epochs_df = pd.DataFrame(extra_epochs_rows)
extra_epochs_df.to_csv(os.path.join(OUT_TABLES, "extra_epochs_needed_extrapolation.csv"), index=False)
median_extra_new = int(np.median(extra_epochs_df["extra_epochs_needed"]))
p75_extra_new = int(np.percentile(extra_epochs_df["extra_epochs_needed"], 75))
max_extra_new = int(extra_epochs_df["extra_epochs_needed"].max())

print(f"NEW run: {n_still_new}/{n_total_new} (method,step) still improving >=3% at epoch 6 (train CE)")
print(f"Median/75th-pct/max extra epochs needed beyond 6: {median_extra_new}/{p75_extra_new}/{max_extra_new}")

# ============================================================================
# Q2 -- DID MORE EPOCHS ACTUALLY HELP? (accuracy comparison)
# ============================================================================
old_acc = old_ranking[old_ranking["method"].isin(SHARED_METHODS)][
    ["method", "first_step", "later_steps", "all_seen"]
].copy()
old_acc = old_acc.rename(columns={"first_step": "first_step_old", "later_steps": "later_steps_old", "all_seen": "all_seen_old"})
old_acc["rank_old_8method_subset"] = old_acc["all_seen_old"].rank(ascending=False).astype(int)

new_acc = new_ranking[["method", "first_step", "later_steps", "all_seen", "rank_all_seen"]].copy()
new_acc = new_acc.rename(columns={"first_step": "first_step_new", "later_steps": "later_steps_new",
                                   "all_seen": "all_seen_new", "rank_all_seen": "rank_new"})

acc_compare = pd.merge(old_acc, new_acc, on="method")
acc_compare["display_name"] = acc_compare["method"].map(dname)
acc_compare["delta_all_seen"] = acc_compare["all_seen_new"] - acc_compare["all_seen_old"]
acc_compare["delta_first_step"] = acc_compare["first_step_new"] - acc_compare["first_step_old"]
acc_compare["delta_later_steps"] = acc_compare["later_steps_new"] - acc_compare["later_steps_old"]
acc_compare["got_worse_all_seen"] = acc_compare["delta_all_seen"] < 0
acc_compare = acc_compare.sort_values("all_seen_new", ascending=False)
acc_compare.to_csv(os.path.join(OUT_TABLES, "accuracy_comparison_old_vs_new.csv"), index=False)

n_worse = int(acc_compare["got_worse_all_seen"].sum())
n_better = int((~acc_compare["got_worse_all_seen"]).sum())
worse_methods = acc_compare[acc_compare["got_worse_all_seen"]][["method", "delta_all_seen"]].values.tolist()
mean_delta = acc_compare["delta_all_seen"].mean()

print(f"Accuracy delta (new-old), all_seen: mean={mean_delta:.2f}pp, {n_better} improved, {n_worse} got worse")
print("Methods that got worse:", worse_methods)

# ============================================================================
# Q3 -- OVERFITTING AUDIT (stricter, 6 epochs / dropout 0.1)
# ============================================================================
of_rows = []
for method in SHARED_METHODS:
    for step in range(1, N_STEPS + 1):
        g = new_hist[(new_hist["method_name"] == method) & (new_hist["step_id"] == step)].sort_values("epoch")
        tce = g["train_ce_loss"].tolist()
        vce = g["val_ce_loss"].tolist()
        ep = g["epoch"].tolist()
        for i, e in enumerate(ep):
            gap = vce[i] - tce[i]
            d_train = (tce[i] - tce[i - 1]) if i > 0 else np.nan
            d_val = (vce[i] - vce[i - 1]) if i > 0 else np.nan
            overfit_sig = bool(i > 0 and d_train < 0 and d_val > 0)
            of_rows.append(dict(
                method_name=method, display_name=dname(method), step_id=step, epoch=e,
                train_ce=tce[i], val_ce=vce[i], train_val_gap=gap,
                epoch_over_epoch_train_delta=d_train, epoch_over_epoch_val_delta=d_val,
                overfitting_signature=overfit_sig,
                overfit_severity=(d_val if overfit_sig else 0.0),
            ))
overfit_table_new = pd.DataFrame(of_rows)
overfit_table_new.to_csv(os.path.join(OUT_TABLES, "overfitting_diagnostic_table_new.csv"), index=False)

n_overfit_new = int(overfit_table_new["overfitting_signature"].sum())
n_transitions_new = int(overfit_table_new["epoch_over_epoch_train_delta"].notna().sum())
overfit_by_method_new = (
    overfit_table_new.groupby("method_name")
    .agg(n_overfit_events=("overfitting_signature", "sum"),
         n_transitions=("epoch_over_epoch_train_delta", lambda s: s.notna().sum()),
         max_severity=("overfit_severity", "max"),
         mean_gap=("train_val_gap", "mean"),
         gap_epoch1=("train_val_gap", "first"),
         final_gap=("train_val_gap", "last"))
    .reset_index()
)
overfit_by_method_new["overfit_event_rate_pct"] = (
    100 * overfit_by_method_new["n_overfit_events"] / overfit_by_method_new["n_transitions"]
)
overfit_by_method_new["gap_growth_epoch1_to_final"] = (
    overfit_by_method_new["final_gap"] - overfit_by_method_new["gap_epoch1"]
)
overfit_by_method_new = overfit_by_method_new.sort_values("n_overfit_events", ascending=False)
overfit_by_method_new.to_csv(os.path.join(OUT_TABLES, "overfitting_summary_by_method_new.csv"), index=False)

# same for OLD (8-method subset) for a like-for-like overfit RATE comparison
of_rows_old = []
for method in SHARED_METHODS:
    for step in range(1, N_STEPS + 1):
        g = old_hist[(old_hist["method_name"] == method) & (old_hist["step_id"] == step)].sort_values("epoch")
        tce = g["train_ce_loss"].tolist()
        vce = g["val_ce_loss"].tolist()
        ep = g["epoch"].tolist()
        for i, e in enumerate(ep):
            d_train = (tce[i] - tce[i - 1]) if i > 0 else np.nan
            d_val = (vce[i] - vce[i - 1]) if i > 0 else np.nan
            overfit_sig = bool(i > 0 and d_train < 0 and d_val > 0)
            of_rows_old.append(dict(method_name=method, step_id=step, epoch=e,
                                     overfitting_signature=overfit_sig,
                                     has_transition=i > 0))
overfit_table_old = pd.DataFrame(of_rows_old)
n_overfit_old = int(overfit_table_old["overfitting_signature"].sum())
n_transitions_old = int(overfit_table_old["has_transition"].sum())

print(f"Overfit-signature rate OLD (8-method subset, 3ep/step): {n_overfit_old}/{n_transitions_old} = "
      f"{100*n_overfit_old/n_transitions_old:.1f}%")
print(f"Overfit-signature rate NEW (8 methods, 6ep/step): {n_overfit_new}/{n_transitions_new} = "
      f"{100*n_overfit_new/n_transitions_new:.1f}%")

# Per-step LOCAL best-epoch (within that step only) -- the meaningful early-stopping question
best_epoch_rows = []
for method in SHARED_METHODS:
    for step in range(1, N_STEPS + 1):
        g = new_hist[(new_hist["method_name"] == method) & (new_hist["step_id"] == step)].sort_values("epoch")
        vce = g["val_ce_loss"].values
        ep = g["epoch"].values
        if len(vce) == 0:
            continue
        best_idx = int(np.argmin(vce))
        best_epoch_rows.append(dict(
            method_name=method, display_name=dname(method), step_id=step,
            best_local_epoch=int(ep[best_idx]), best_val_ce=vce[best_idx],
            final_local_epoch=int(ep[-1]), final_val_ce=vce[-1],
            best_epoch_lt_final=bool(ep[best_idx] < ep[-1]),
            val_ce_final_minus_best=vce[-1] - vce[best_idx],
        ))
best_epoch_df = pd.DataFrame(best_epoch_rows)
best_epoch_df.to_csv(os.path.join(OUT_TABLES, "per_step_local_best_epoch_new.csv"), index=False)
n_best_lt_final = int(best_epoch_df["best_epoch_lt_final"].sum())
n_best_total = len(best_epoch_df)
print(f"Per-step LOCAL best-epoch < final(6) in {n_best_lt_final}/{n_best_total} (method,step) cases")

# ============================================================================
# Q4 -- RANKING STABILITY
# ============================================================================
rank_compare = acc_compare[["method", "display_name", "rank_old_8method_subset", "rank_new",
                             "all_seen_old", "all_seen_new", "delta_all_seen"]].copy()
rank_compare["rank_change"] = rank_compare["rank_old_8method_subset"] - rank_compare["rank_new"]
rank_compare = rank_compare.sort_values("rank_new")
rank_compare.to_csv(os.path.join(OUT_TABLES, "ranking_comparison_old_vs_new.csv"), index=False)

spearman_rho, spearman_p = spearmanr(rank_compare["rank_old_8method_subset"], rank_compare["rank_new"])
n_rank_changes = int((rank_compare["rank_change"] != 0).sum())
top2_old = old_acc.sort_values("all_seen_old", ascending=False)["method"].tolist()[:2]
top2_new = new_ranking.sort_values("all_seen", ascending=False)["method"].tolist()[:2]
top2_stable = top2_old == top2_new

print(f"Spearman rank correlation (old-8-subset vs new): rho={spearman_rho:.3f}, p={spearman_p:.4f}")
print(f"Rank changes: {n_rank_changes}/8 methods changed rank; top-2 stable: {top2_stable}")
print("OLD top-2:", top2_old, "NEW top-2:", top2_new)

# ============================================================================
# FIGURES
# ============================================================================

# ---- Fig 1: grouped bar chart, all_seen accuracy old vs new, 8 shared methods ----
fig, ax = plt.subplots(figsize=(9.5, 5.8))
order = acc_compare.sort_values("all_seen_new", ascending=False)
x = np.arange(len(order))
w = 0.36
ax.bar(x - w / 2, order["all_seen_old"], width=w, label="OLD (3 epochs, dropout 0.05)", color="#9e9e9e")
ax.bar(x + w / 2, order["all_seen_new"], width=w, label="NEW (6 epochs, dropout 0.1, best-epoch sel.)",
       color="#1f77b4")
for i, (o, n) in enumerate(zip(order["all_seen_old"], order["all_seen_new"])):
    d = n - o
    color = "crimson" if d < 0 else "seagreen"
    ax.text(i, max(o, n) + 1.2, f"{d:+.1f}", ha="center", fontsize=8, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([dname(m) for m in order["method"]], rotation=35, ha="right")
ax.set_ylabel("all_seen accuracy (%)")
ax.set_title("Final all_seen accuracy: OLD vs NEW run (8 shared methods)\n(label = delta pp, new $-$ old; red = regression)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, "1_accuracy_old_vs_new.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: final-epoch relative train-CE improvement, NEW run, per (method,step) ----
fig, ax = plt.subplots(figsize=(10.5, 6.0))
piv = new_verdict.pivot(index="method_name", columns="step_id", values="final_epoch_rel_train_improvement_pct")
piv = piv.reindex(SHARED_METHODS)
im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto", vmin=-5, vmax=30)
ax.set_xticks(range(N_STEPS))
ax.set_xticklabels([f"step {s}" for s in range(1, N_STEPS + 1)])
ax.set_yticks(range(len(piv)))
ax.set_yticklabels([dname(m) for m in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                     color="white" if abs(v) > 18 else "black")
ax.axvline(-0.5, color="black", lw=0.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("relative train-CE improvement at epoch 6 vs epoch 5 (%)")
ax.set_title("NEW run: is training still improving materially at epoch 6?\n(>=3% = still under-trained; <3% = converged; dashed line at 3% is the threshold)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, "2_convergence_heatmap_new_run.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: train/val CE gap trajectory across 6 epochs, all methods (global epoch axis) ----
fig, ax = plt.subplots(figsize=(10, 6.0))
for method in SHARED_METHODS:
    g = overfit_table_new[overfit_table_new["method_name"] == method].sort_values(["step_id", "epoch"])
    global_epoch = np.arange(1, len(g) + 1)
    lw = 2.2 if method in top2_new else 1.2
    alpha = 1.0 if method in top2_new else 0.7
    ax.plot(global_epoch, g["train_val_gap"], color=METHOD_COLOR[method], lw=lw, alpha=alpha,
            marker="o", ms=2.5, label=dname(method))
for s in range(1, N_STEPS):
    ax.axvline(s * NEW_EPOCHS_PER_STEP + 0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("global epoch (dotted = CL step boundary; 6 epochs/step)")
ax.set_ylabel("val CE $-$ train CE  (generalization gap)")
ax.set_title("NEW run: train/val CE gap across 6 epochs/step, dropout=0.1\n(rising gap = overfitting signature)")
outside_legend(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, "3_train_val_gap_new_run.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: ranking slope chart, old (8-subset) vs new ----
fig, ax = plt.subplots(figsize=(7.5, 6.5))
for _, r in rank_compare.iterrows():
    ax.plot([0, 1], [r["rank_old_8method_subset"], r["rank_new"]], marker="o", ms=7,
            color=METHOD_COLOR[r["method"]], lw=2.2)
    ax.text(-0.05, r["rank_old_8method_subset"], dname(r["method"]), ha="right", va="center", fontsize=8.5)
    ax.text(1.05, r["rank_new"], f"#{int(r['rank_new'])}", ha="left", va="center", fontsize=8.5)
ax.set_xlim(-0.6, 1.3)
ax.set_ylim(8.7, 0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["OLD\n(8-method subset)", "NEW"])
ax.set_ylabel("rank by all_seen accuracy (1 = best)")
ax.set_title(f"Ranking stability, OLD vs NEW (Spearman $\\rho$={spearman_rho:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, "4_ranking_stability_slopechart.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 5: per-step local best-epoch vs final-epoch val CE ----
fig, ax = plt.subplots(figsize=(10.5, 6.0))
bep = best_epoch_df.copy()
bep["label"] = bep["display_name"] + " / step" + bep["step_id"].astype(str)
bep = bep.sort_values(["method_name", "step_id"])
x = np.arange(len(bep))
colors = ["crimson" if v else "steelblue" for v in bep["best_epoch_lt_final"]]
ax.bar(x, bep["val_ce_final_minus_best"], color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(bep["label"], rotation=90, fontsize=6.5)
ax.set_ylabel("final-epoch val CE $-$ best-local-epoch val CE")
ax.set_title("NEW run: within-step best-epoch vs final(6th)-epoch val CE, all (method,step)\n"
             f"(red = best epoch < 6, i.e. best-epoch selection would trigger; {n_best_lt_final}/{n_best_total} cases)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_PLOTS, "5_best_epoch_vs_final_epoch.png"), bbox_inches="tight")
plt.close(fig)

print("All figures and tables written.")

# ============================================================================
# Save summary stats for the report writer
# ============================================================================
import json
summary_stats = dict(
    n_total_new=int(n_total_new), n_still_new=int(n_still_new), n_converged_new=int(n_converged_new),
    n_diverging_new=int(n_diverging_new), mean_rel_new=float(mean_rel_new),
    n_total_old=int(n_total_old), n_still_old=int(n_still_old),
    n_val_still_new=int(n_val_still_new), n_val_converged_new=int(n_val_converged_new),
    n_val_diverging_new=int(n_val_diverging_new),
    median_extra_new=median_extra_new, p75_extra_new=p75_extra_new, max_extra_new=max_extra_new,
    n_worse=n_worse, n_better=n_better, mean_delta=float(mean_delta),
    n_overfit_new=n_overfit_new, n_transitions_new=n_transitions_new,
    n_overfit_old=n_overfit_old, n_transitions_old=n_transitions_old,
    n_best_lt_final=n_best_lt_final, n_best_total=n_best_total,
    spearman_rho=float(spearman_rho), spearman_p=float(spearman_p),
    n_rank_changes=n_rank_changes, top2_stable=bool(top2_stable),
    top2_old=top2_old, top2_new=top2_new,
)
with open(os.path.join(OUT_TABLES, "_summary_stats.json"), "w") as f:
    json.dump(summary_stats, f, indent=2)
print(json.dumps(summary_stats, indent=2))
