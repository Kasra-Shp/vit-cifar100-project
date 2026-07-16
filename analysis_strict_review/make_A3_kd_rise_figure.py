"""
A3 documentation figure: for rank_extension_kd_only_T2 in the NEW (revert) run,
show the within-step KD-loss RISE aligned with the same-epoch train-CE DROP,
using a twin (dual) y-axis so the timing match is directly visible. This is
visual evidence for the already-verified structural explanation (old, frozen
rank blocks stay active in the forward pass while the new block trains into
the SAME shared weight, so KD divergence rises exactly as CE-driven new-class
fitting is steepest, before both settle for the rest of the step) -- not a new
claim, just its documentation figure. No training; reads only the saved
tables/training_loss_history_by_epoch.csv from the NEW run.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

NEW = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/R3/results_revert_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_193538")
OUT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_review")

LORA_EPOCHS = 9
METHOD = "rank_extension_kd_only_T2"

e = pd.read_csv(NEW / "tables" / "training_loss_history_by_epoch.csv")
sub = e[e.method == METHOD].sort_values(["cl_step", "local_epoch"]).copy()
sub["global_epoch"] = (sub["cl_step"] - 1) * LORA_EPOCHS + sub["local_epoch"]

fig, ax1 = plt.subplots(figsize=(13, 6))
ax2 = ax1.twinx()

l1, = ax1.plot(sub.global_epoch, sub.train_ce_loss, color="#1f77b4", lw=2.2, marker="o", ms=3, label="Train CE (left axis)")
l2, = ax2.plot(sub.global_epoch, sub.kd_loss_weighted, color="#d62728", lw=2.2, marker="s", ms=3, label="KD weighted loss (right axis)")

for st in range(1, 6):
    b = (st - 1) * LORA_EPOCHS
    ax1.axvline(b + 0.5, color="#bbb", linestyle=":", lw=1)
# mark the epoch1->epoch2 rise window of each step transition (>=step 2)
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
    "rank_extension_kd_only_T2 (NEW/revert run) -- within-step KD rise aligned with the same-epoch CE drop",
    fontsize=14, fontweight="bold",
)
ax1.set_title(
    "At every step transition (S2-S5), KD RISES from epoch 1 to epoch 2 (+26% to +35%) in the exact\n"
    "same epoch where train CE drops fastest (-73% to -78%) -- old, frozen rank blocks stay active in the\n"
    "forward pass while the new block's rapid CE-driven fitting perturbs the SAME shared weight, pulling\n"
    "the combined output away from the frozen (previous-step) KD teacher before both re-converge.",
    fontsize=10, loc="left",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
outpath = OUT / "A3_rankext_kd_rise_vs_ce_drop.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", outpath)

# print the numeric alignment table used to caption the figure
rows = []
for st in [2, 3, 4, 5]:
    s = sub[sub.cl_step == st].sort_values("local_epoch")
    e1 = s[s.local_epoch == 1].iloc[0]
    e2 = s[s.local_epoch == 2].iloc[0]
    rows.append({
        "step": st,
        "kd_epoch1": e1.kd_loss_weighted, "kd_epoch2": e2.kd_loss_weighted,
        "kd_rise_pct": 100 * (e2.kd_loss_weighted - e1.kd_loss_weighted) / e1.kd_loss_weighted,
        "ce_epoch1": e1.train_ce_loss, "ce_epoch2": e2.train_ce_loss,
        "ce_drop_pct": 100 * (e2.train_ce_loss - e1.train_ce_loss) / e1.train_ce_loss,
    })
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "A3_kd_rise_ce_drop_alignment.csv", index=False)
print(tab.to_string(index=False))
