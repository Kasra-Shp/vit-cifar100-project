"""
Regenerates plots/combined_loss_decomposition.png (redesigned, one-panel-per-
variant layout) for the STRICT/instrumented run (20260717_015946), from the
saved tables/training_loss_history_by_epoch.csv -- no training. Also verifies
the TOTAL == sum-of-parts identity on THIS run's data, and re-confirms that
for the Base variant, train_ce_loss == train_total_loss exactly (KD=orth=0).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

STRICT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/R3/results_strict_20260717_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260717_015946")
OUT = Path(r"C:/Users/ASUSCenter/Desktop/vit-cifar100-project/analysis_strict_run")

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

E = pd.read_csv(STRICT / "tables" / "training_loss_history_by_epoch.csv")
E["variant"] = E["method"].map(VARIANT)
E["family"] = E["method"].map(lambda m: "simple_avg" if str(m).startswith("simple_avg") else "rank_extension")
method_by_family_variant = {(fam, v): m for m, v in VARIANT.items() for fam in [("simple_avg" if m.startswith("simple_avg") else "rank_extension")]}

line_specs = [
    ("train_ce_loss", "Train CE", "#1f77b4", "-", 2.0),
    ("kd_loss_weighted", "KD weighted", "#ff7f0e", "--", 2.0),
    ("factor_orth_loss_weighted", "Factor-Orth weighted", "#d62728", ":", 2.0),
    ("train_total_loss", "TOTAL (= sum of the above)", "#000000", "-", 3.0),
]

fig, axs = plt.subplots(len(SUPERVISOR_VARIANT_ORDER), 2, figsize=(16, 15), sharex=True)
fig.suptitle(
    "Combined Loss Decomposition -- STRICT (instrumented) run, 20260717_015946 -- per-variant panels\n"
    "(all lines share ONE log-scale y-axis per panel; TOTAL is plotted directly, never a separate scale,\n"
    "so 'TOTAL = sum of the other lines' is checkable by eye within each panel)",
    fontsize=15, fontweight="bold", y=.995,
)

for rr, v in enumerate(SUPERVISOR_VARIANT_ORDER):
    for cc, f in enumerate(FAMS):
        ax = axs[rr, cc]
        m = method_by_family_variant.get((f, v))
        if rr == 0:
            ax.set_title(FLAB[f], fontweight="bold")
        if cc == 0:
            ax.set_ylabel(f"{v}\n(log scale)")
        ax.grid(True, axis="y", color="#ddd")
        ax.set_yscale("log")
        for b in range(LORA_EPOCHS, NUM_STEPS * LORA_EPOCHS, LORA_EPOCHS):
            ax.axvline(b + .5, color="#bbb", linestyle=":", lw=1)
        fd = E[(E.family == f) & (E.method == m)] if m is not None else E.iloc[0:0]
        for st in range(1, NUM_STEPS + 1):
            s = fd[fd.cl_step == st].sort_values("local_epoch")
            if len(s) == 0:
                continue
            x = (st - 1) * LORA_EPOCHS + s.local_epoch.astype(float)
            for met, lab, color, ls, lw in line_specs:
                y = pd.to_numeric(s.get(met, np.nan), errors="coerce")
                good = np.isfinite(y) & (y > 0)
                if good.any():
                    ax.plot(x[good], y[good], color=color, linestyle=ls, lw=lw)
        ax.set_xticks([(i * LORA_EPOCHS) + 2 for i in range(NUM_STEPS)])
        ax.set_xticklabels([f"S{i}" for i in range(1, NUM_STEPS + 1)])

legend_handles = [Line2D([0], [0], color=color, linestyle=ls, lw=max(lw, 2.5)) for _, lab, color, ls, lw in line_specs]
legend_labels = [lab for _, lab, _, _, _ in line_specs]
fig.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(.915, .52), frameon=False, title="Line (all 4 on\nthe same axis)")
fig.tight_layout(rect=[.02, .02, .90, .93])
outpath = OUT / "Q2_combined_loss_decomposition_STRICT_run_redesigned.png"
plt.savefig(outpath, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", outpath)

# ---- verify Total == sum of parts within the STRICT run's data ----
recon = (E["train_ce_loss"].fillna(0) + E["kd_loss_weighted"].fillna(0)
         + E["factor_orth_loss_weighted"].fillna(0) + E["delta_trace_loss_weighted"].fillna(0))
mismatch = (recon - E["train_total_loss"]).abs().max()
print("Max |reconstructed_total - reported_total| (STRICT run, all methods/epochs):", mismatch)

# ---- verify Base identity: train_ce_loss == train_total_loss exactly for Base variant ----
base_rows = E[E.variant == "Base"]
diff = (base_rows["train_ce_loss"] - base_rows["train_total_loss"]).abs()
print(f"Base variant (n={len(base_rows)} rows, both families, all steps/epochs): "
      f"max |train_ce_loss - train_total_loss| = {diff.max():.3e}")
print("kd_loss_weighted all zero/NaN for Base?", base_rows["kd_loss_weighted"].fillna(0).abs().max())
print("factor_orth_loss_weighted all zero/NaN for Base?", base_rows["factor_orth_loss_weighted"].fillna(0).abs().max())
