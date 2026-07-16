"""
Regenerates plots/combined_loss_decomposition.png for the NEW (revert) run,
from the already-saved tables/training_loss_history_by_epoch.csv -- no
training. Mirrors the FIXED plotting logic now in
vit_lora_cifar100_full5step_n5.py (the "SUPERVISOR FIX (comment 2)" blocks):
adds a legend and a linear/log annotation per row that the original figure
in both OLD and NEW runs was missing.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

NEW = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/R3/results_revert_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_193538")
OUT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_revert_run")

LORA_EPOCHS = 9
NUM_STEPS = 5
FAMS = ["simple_avg", "rank_extension"]
FLAB = {"simple_avg": "Simple-Average Family", "rank_extension": "Rank-Extension Family"}
SUPERVISOR_VARIANT_ORDER = ["Base", "KD (T=2)", "Factor-Orth", "KD + Factor-Orth"]
VARIANT = {
    "simple_avg": "Base", "rank_extension": "Base",
    "simple_avg_factor_orth": "Factor-Orth", "rank_extension_orth_factor_lam_50": "Factor-Orth",
    "simple_avg_kd_T2": "KD (T=2)", "rank_extension_kd_only_T2": "KD (T=2)",
    "simple_avg_factor_orth_kd_T2": "KD + Factor-Orth", "rank_extension_orth_factor_lam_50_kd_T2": "KD + Factor-Orth",
}
VCOL = {"Base": "#1f77b4", "KD (T=2)": "#ff7f0e", "Factor-Orth": "#d62728", "KD + Factor-Orth": "#2ca02c"}
VSTYLE = {"Base": "-", "KD (T=2)": "--", "Factor-Orth": ":", "KD + Factor-Orth": "-."}

E = pd.read_csv(NEW / "tables" / "training_loss_history_by_epoch.csv")
E["variant"] = E["method"].map(VARIANT)
E["family"] = E["method"].map(lambda m: "simple_avg" if str(m).startswith("simple_avg") else "rank_extension")

comps = [
    ("train_ce_loss", "Train CE", False),
    ("kd_loss_weighted", "KD weighted", False),
    ("factor_orth_loss_weighted", "Factor-Orth weighted", True),
    ("train_total_loss", "Total train loss", True),
]

fig, axs = plt.subplots(len(comps), 2, figsize=(16, 13), sharex=True)
fig.suptitle("Combined Loss Decomposition -- NEW (revert) run, 20260716_193538", fontsize=20, fontweight="bold", y=.995)

for rr, (met, lab, logy) in enumerate(comps):
    for cc, f in enumerate(FAMS):
        ax = axs[rr, cc]
        fd = E[E.family == f]
        if rr == 0:
            ax.set_title(FLAB[f], fontweight="bold")
        if cc == 0:
            ax.set_ylabel(lab + ("\n(log scale)" if logy else "\n(linear scale)"))
        ax.grid(True, axis="y", color="#ddd")
        if logy:
            ax.set_yscale("log")
        for b in range(LORA_EPOCHS, NUM_STEPS * LORA_EPOCHS, LORA_EPOCHS):
            ax.axvline(b + .5, color="#bbb", linestyle=":", lw=1)
        for st in range(1, NUM_STEPS + 1):
            for v in SUPERVISOR_VARIANT_ORDER:
                s = fd[(fd.cl_step == st) & (fd.variant == v)].sort_values("local_epoch")
                y = pd.to_numeric(s.get(met, np.nan), errors="coerce")
                good = np.isfinite(y) & ((y > 0) if logy else True)
                x = (st - 1) * LORA_EPOCHS + s.local_epoch.astype(float)
                if len(s) > 0 and good.any():
                    ax.plot(x[good], y[good], color=VCOL[v], linestyle=VSTYLE[v], lw=2.1)
        ax.set_xticks([(i * LORA_EPOCHS) + 2 for i in range(NUM_STEPS)])
        ax.set_xticklabels([f"S{i}" for i in range(1, NUM_STEPS + 1)])

legend_handles = [Line2D([0], [0], color=VCOL[v], linestyle=VSTYLE[v], lw=3) for v in SUPERVISOR_VARIANT_ORDER]
fig.legend(legend_handles, SUPERVISOR_VARIANT_ORDER, loc="center left", bbox_to_anchor=(.915, .52), frameon=False, title="Variant")
fig.tight_layout(rect=[.02, .02, .90, .95])
outpath = OUT / "combined_loss_decomposition_NEW_run_legend_fixed.png"
plt.savefig(outpath, dpi=220, bbox_inches="tight")
plt.close()
print("Saved:", outpath)

# ---- numeric check: does the "blue bend" persist in the NEW run? ----
blue_simple = E[(E.method == "simple_avg")]
blue_rankext = E[(E.method == "rank_extension")]
print("\nSimpleAvg Base (blue) train_total_loss range (NEW run): min=%.4f max=%.4f (%.1fx)" % (
    blue_simple.train_total_loss.min(), blue_simple.train_total_loss.max(),
    blue_simple.train_total_loss.max() / blue_simple.train_total_loss.min()))
fam_simple = E[E.family == "simple_avg"]
print("SimpleAvg family shared total_loss axis max (NEW run): %.2f (driven by %s)" % (
    fam_simple.train_total_loss.max(),
    fam_simple.loc[fam_simple.train_total_loss.idxmax(), "method"]))

print("\nRankExt Base (blue) train_total_loss range (NEW run): min=%.4f max=%.4f (%.1fx)" % (
    blue_rankext.train_total_loss.min(), blue_rankext.train_total_loss.max(),
    blue_rankext.train_total_loss.max() / blue_rankext.train_total_loss.min()))
fam_rankext = E[E.family == "rank_extension"]
print("RankExt family shared total_loss axis max (NEW run): %.2f (driven by %s)" % (
    fam_rankext.train_total_loss.max(),
    fam_rankext.loc[fam_rankext.train_total_loss.idxmax(), "method"]))

# bookkeeping-bug check on NEW run too
recon = (E["train_ce_loss"].fillna(0) + E["kd_loss_weighted"].fillna(0)
         + E["factor_orth_loss_weighted"].fillna(0) + E["delta_trace_loss_weighted"].fillna(0))
mismatch = (recon - E["train_total_loss"]).abs().max()
print("\nMax |reconstructed_total - reported_total| across all methods/epochs (NEW run):", mismatch)
