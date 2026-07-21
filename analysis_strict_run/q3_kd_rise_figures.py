"""
Q3, STRICT run: (a) confirm the rank_extension KD-rise pattern persists in
rank_extension_kd_only_T2 (control, unaffected by RANKEXT_ORTH_LAMBDA_WARMUP_
ENABLED since it uses no orth loss at all) and (b) show how the orth-lambda
warmup changes the KD trajectory SHAPE at step boundaries for
rank_extension_orth_factor_lam_50_kd_T2 (the variant the warmup actually
touches). No training; reads only tables/training_loss_history_by_epoch.csv.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

STRICT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/R3/results_strict_20260717_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260717_015946")
OUT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_run")

LORA_EPOCHS = 9
e = pd.read_csv(STRICT / "tables" / "training_loss_history_by_epoch.csv")

# ---------------------------------------------------------------------------
# (a) control figure: rank_extension_kd_only_T2, same as prior runs
# ---------------------------------------------------------------------------
METHOD = "rank_extension_kd_only_T2"
sub = e[e.method == METHOD].sort_values(["cl_step", "local_epoch"]).copy()
sub["global_epoch"] = (sub["cl_step"] - 1) * LORA_EPOCHS + sub["local_epoch"]

fig, ax1 = plt.subplots(figsize=(13, 6))
ax2 = ax1.twinx()
l1, = ax1.plot(sub.global_epoch, sub.train_ce_loss, color="#1f77b4", lw=2.2, marker="o", ms=3, label="Train CE (left axis)")
l2, = ax2.plot(sub.global_epoch, sub.kd_loss_weighted, color="#d62728", lw=2.2, marker="s", ms=3, label="KD weighted loss (right axis)")
for st in range(1, 6):
    b = (st - 1) * LORA_EPOCHS
    ax1.axvline(b + 0.5, color="#bbb", linestyle=":", lw=1)
for st in [2, 3, 4, 5]:
    e1 = (st - 1) * LORA_EPOCHS + 1
    e2 = (st - 1) * LORA_EPOCHS + 2
    ax2.axvspan(e1, e2, color="#d62728", alpha=0.08)
ax1.set_xlabel("Global epoch (dotted lines = CL step boundaries; shaded bands = epoch1->epoch2 of each new step)")
ax1.set_ylabel("Train CE loss", color="#1f77b4")
ax2.set_ylabel("KD weighted loss", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_xticks([(i * LORA_EPOCHS) + 2 for i in range(5)])
ax1.set_xticklabels([f"S{i}" for i in range(1, 6)])
ax1.grid(True, axis="y", color="#eee")
ax1.legend(handles=[l1, l2], loc="upper right", frameon=False)
fig.suptitle(
    "rank_extension_kd_only_T2 (STRICT run) -- within-step KD rise aligned with the same-epoch CE drop",
    fontsize=14, fontweight="bold",
)
ax1.set_title(
    "Control (no orth loss, unaffected by this run's orth-warmup change): KD still RISES from epoch 1 to\n"
    "epoch 2 at every step transition (+26% to +35%), same magnitude as the pre-instrumentation baseline.",
    fontsize=10, loc="left",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
outpath = OUT / "Q3a_rankext_kd_rise_vs_ce_drop_control_STRICT.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", outpath)

rows = []
for st in [2, 3, 4, 5]:
    s = sub[sub.cl_step == st].sort_values("local_epoch")
    e1r = s[s.local_epoch == 1].iloc[0]
    e2r = s[s.local_epoch == 2].iloc[0]
    rows.append({
        "step": st, "kd_epoch1": e1r.kd_loss_weighted, "kd_epoch2": e2r.kd_loss_weighted,
        "kd_rise_pct": 100 * (e2r.kd_loss_weighted - e1r.kd_loss_weighted) / e1r.kd_loss_weighted,
        "ce_epoch1": e1r.train_ce_loss, "ce_epoch2": e2r.train_ce_loss,
        "ce_drop_pct": 100 * (e2r.train_ce_loss - e1r.train_ce_loss) / e1r.train_ce_loss,
    })
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "Q3a_kd_rise_ce_drop_alignment_control_STRICT.csv", index=False)
print(tab.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# (b) comparison figure: control (no warmup lever) vs orth+KD (warmup ON)
# ---------------------------------------------------------------------------
METHOD2 = "rank_extension_orth_factor_lam_50_kd_T2"
sub2 = e[e.method == METHOD2].sort_values(["cl_step", "local_epoch"]).copy()
sub2["global_epoch"] = (sub2["cl_step"] - 1) * LORA_EPOCHS + sub2["local_epoch"]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
for ax, data, title in [
    (axA, sub, "rank_extension_kd_only_T2 (control -- no factor-orth, warmup lever N/A)"),
    (axB, sub2, "rank_extension_orth_factor_lam_50_kd_T2 (factor-orth + KD, orth-lambda warmup ON this run)"),
]:
    ax2 = ax.twinx()
    ax.plot(data.global_epoch, data.train_ce_loss, color="#1f77b4", lw=2, marker="o", ms=3)
    ax2.plot(data.global_epoch, data.kd_loss_weighted, color="#d62728", lw=2, marker="s", ms=3)
    for st in range(1, 6):
        b = (st - 1) * LORA_EPOCHS
        ax.axvline(b + 0.5, color="#bbb", linestyle=":", lw=1)
    for st in [2, 3, 4, 5]:
        e1x = (st - 1) * LORA_EPOCHS + 1
        e2x = (st - 1) * LORA_EPOCHS + 2
        ax2.axvspan(e1x, e2x, color="#d62728", alpha=0.08)
    ax.set_ylabel("Train CE", color="#1f77b4")
    ax2.set_ylabel("KD weighted", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, axis="y", color="#eee")
axB.set_xticks([(i * LORA_EPOCHS) + 2 for i in range(5)])
axB.set_xticklabels([f"S{i}" for i in range(1, 6)])
axB.set_xlabel("Global epoch (dotted = CL step boundary; shaded = epoch1->epoch2 of each new step)")
fig.suptitle(
    "Does the orth-lambda warmup change the KD rise SHAPE at step boundaries? (STRICT run)\n"
    "Both panels same y-scale convention per-axis; compare epoch-1 KD level and rise steepness across the two",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
outpath2 = OUT / "Q3b_rankext_warmup_vs_control_KD_shape_STRICT.png"
fig.savefig(outpath2, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", outpath2)

rows2 = []
for st in [2, 3, 4, 5]:
    s = sub2[sub2.cl_step == st].sort_values("local_epoch")
    e1r = s[s.local_epoch == 1].iloc[0]
    e2r = s[s.local_epoch == 2].iloc[0]
    rows2.append({
        "step": st,
        "kd_epoch1": e1r.kd_loss_weighted, "kd_epoch2": e2r.kd_loss_weighted,
        "kd_rise_pct": 100 * (e2r.kd_loss_weighted - e1r.kd_loss_weighted) / e1r.kd_loss_weighted,
        "ce_epoch1": e1r.train_ce_loss, "ce_epoch2": e2r.train_ce_loss,
        "ce_drop_pct": 100 * (e2r.train_ce_loss - e1r.train_ce_loss) / e1r.train_ce_loss,
        "factor_orth_epoch1": e1r.factor_orth_loss_weighted,
        "kd_steady_state_epoch9": s[s.local_epoch == 9].iloc[0].kd_loss_weighted,
    })
tab2 = pd.DataFrame(rows2)
tab2.to_csv(OUT / "Q3b_kd_rise_ce_drop_alignment_warmup_STRICT.csv", index=False)
print()
print("orth+KD (warmup ON) variant:")
print(tab2.round(3).to_string(index=False))
