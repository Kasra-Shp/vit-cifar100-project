import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10.5,
})

B_DIR = "R3/results_EPOCH6_20260711_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260711_214100"
N_DIR = "R3/results_calibfix_20260716_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260716_133338"
OUT = "analysis_rankext_firststep/"

DISPLAY = {
    "rank_extension": "RankExt",
    "rank_extension_kd_only_T2": "RankExt+KD",
    "rank_extension_orth_factor_lam_50": "RankExt+FactorOrth",
    "rank_extension_orth_factor_lam_50_kd_T2": "RankExt+FactorOrth+KD",
}
RANKEXT_METHODS = list(DISPLAY.keys())

table2 = pd.read_csv(f"{OUT}table2_rankext_perstep_and_forgetting.csv")
table3 = pd.read_csv(f"{OUT}table3_step1_trainingtime_vs_evaltime.csv")
b_hist = pd.read_csv(f"{B_DIR}/tables/training_loss_history_by_epoch.csv")
n_hist = pd.read_csv(f"{N_DIR}/tables/training_loss_history_by_epoch.csv")
table5 = pd.read_csv(f"{OUT}table5_rankext_vs_rankext_orth_headtohead.csv")

# ============================================================
# Figure 1: first_step accuracy delta bar chart, 4 rank_ext variants
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 4.6))
t2 = table2.sort_values("delta_first_step_pp")
colors = [RED if v < 0 else BLUE for v in t2.delta_first_step_pp]
bars = ax.barh(t2.display_name, t2.delta_first_step_pp, color=colors, height=0.55, zorder=3)
ax.axvline(0, color=INK, linewidth=1)
for bar, v, b, n in zip(bars, t2.delta_first_step_pp, t2.BASELINE_first_step_acc, t2.NEW_first_step_acc):
    x = bar.get_width()
    ax.text(x + (0.6 if x >= 0 else -0.6), bar.get_y() + bar.get_height() / 2,
            f"{v:+.1f} pp  ({b:.1f}%→{n:.1f}%)", va="center",
            ha="left" if x >= 0 else "right", fontsize=9.5, color=INK)
ax.set_title("first_step accuracy delta: BASELINE (20260711) → NEW (20260716 calibfix)",
             fontsize=11.5, color=INK, pad=14, loc="left")
ax.set_xlabel("Δ first_step accuracy (percentage points)")
ax.set_xlim(-22, 22)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.savefig(OUT + "fig1_first_step_accuracy_deltas.png", dpi=160)
plt.close()

# ============================================================
# Figure 2: step-1 training-time val CE overlay (left) + training-time
#           vs final post-all-steps eval loss "inflation" (right)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# left panel: step-1 val CE trajectory, BASELINE (6ep) vs NEW (9ep).
# step-1 training is identical across all 4 rank_ext variants within a run
# (orth/KD only diverge from step 2 onward), so one representative curve per run.
b_step1 = b_hist[(b_hist.method == "rank_extension") & (b_hist.cl_step == 1)].sort_values("local_epoch")
n_step1 = n_hist[(n_hist.method == "rank_extension") & (n_hist.cl_step == 1)].sort_values("local_epoch")
axes[0].plot(b_step1.local_epoch, b_step1.val_ce_loss, "o--", color=ORANGE, linewidth=2, label="BASELINE (6 epochs, dropout 0.10)")
axes[0].plot(n_step1.local_epoch, n_step1.val_ce_loss, "o-", color=BLUE, linewidth=2, label="NEW (9 epochs, dropout 0.05)")
axes[0].set_xlabel("epoch within CL step 1")
axes[0].set_ylabel("validation CE loss (step-1 val set)")
axes[0].set_title("Step-1 training is healthy & near-identical\nacross all 4 rank_ext variants in both runs", fontsize=11, color=INK, loc="left")
axes[0].legend(frameon=False, fontsize=9.5)
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(0, max(b_step1.val_ce_loss.max(), n_step1.val_ce_loss.max()) * 1.3)

# right panel: bar chart of final post-all-steps first_step eval loss per method per run,
# with reference lines at the step-1 training-time value.
x = range(len(RANKEXT_METHODS))
width = 0.32
b_vals = [table3[table3.method == m].BASELINE_step1_FINAL_eval_loss_post_all_steps.iloc[0] for m in RANKEXT_METHODS]
n_vals = [table3[table3.method == m].NEW_step1_FINAL_eval_loss_post_all_steps.iloc[0] for m in RANKEXT_METHODS]
axes[1].bar([i - width / 2 for i in x], b_vals, width=width, color=ORANGE, label="BASELINE final eval loss", zorder=3)
axes[1].bar([i + width / 2 for i in x], n_vals, width=width, color=BLUE, label="NEW final eval loss", zorder=3)
axes[1].axhline(0.146, color=ORANGE, linestyle=":", linewidth=1.3, label="BASELINE step-1 training-time val CE (≈0.15)")
axes[1].axhline(0.166, color=BLUE, linestyle=":", linewidth=1.3, label="NEW step-1 training-time val CE (≈0.17)")
for i, (bv, nv) in enumerate(zip(b_vals, n_vals)):
    axes[1].text(i - width / 2, bv + 0.15, f"{bv:.1f}", ha="center", fontsize=8.5, color=INK2)
    axes[1].text(i + width / 2, nv + 0.15, f"{nv:.1f}", ha="center", fontsize=8.5, color=INK)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([DISPLAY[m] for m in RANKEXT_METHODS], fontsize=9, rotation=12)
axes[1].set_ylabel("first_step eval CE loss (post all 5 steps)")
axes[1].set_title("Final first_step eval loss is 5–69× worse than\nstep-1's own training-time loss → forgetting-time, not training-time",
                   fontsize=11, color=INK, loc="left")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT + "fig2_step1_trainingtime_vs_forgetting.png", dpi=160, bbox_inches="tight")
plt.close()

# ============================================================
# Figure 3: factor-orth loss trajectories by family (rank_ext+orth vs simple_avg+orth)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
family_methods = {
    "rank_extension_orth_factor_lam_50": ("RankExt+FactorOrth", 0),
    "simple_avg_factor_orth": ("SimpleAvg+FactorOrth", 1),
}

for row_idx, (label, hist) in enumerate([("BASELINE", b_hist), ("NEW", n_hist)]):
    for m, (name, col) in family_methods.items():
        ax = axes[row_idx, col]
        sub = hist[hist.method == m].sort_values(["cl_step", "local_epoch"]).reset_index(drop=True)
        sub = sub[sub.cl_step > 1]  # orth only active from step 2 onward
        xvals = range(len(sub))
        ax.plot(xvals, sub.factor_orth_loss_weighted, color=VIOLET, linewidth=1.8, zorder=3)
        ax.set_yscale("log")
        boundaries = sub.groupby("cl_step").size().cumsum().values[:-1]
        for b in boundaries:
            ax.axvline(b - 0.5, color=GRID, linewidth=1, zorder=1)
        ax.set_title(f"{label}: {name}", fontsize=10.5, color=INK, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if row_idx == 1:
            ax.set_xlabel("epoch index (concatenated across CL steps 2→5)")
        if col == 0:
            ax.set_ylabel("factor_orth_loss_weighted (log scale)")

fig.suptitle("factor-orth penalty is ~20–300× smaller (relative to CE) for rank_ext than for simple_avg,\n"
             "in both runs — the new rank block starts close to orthogonal, so the penalty is quickly trivially satisfied",
             fontsize=11.5, color=INK2, y=1.03)
plt.tight_layout()
plt.savefig(OUT + "fig3_factororth_trajectories_by_family.png", dpi=160, bbox_inches="tight")
plt.close()

# ============================================================
# Figure 4: rank_ext vs rank_ext+orth accuracy comparison (plain vs orth), 2x2 (run x kd_state)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
groups = ["first_step", "later_steps", "all_seen"]
xg = range(len(groups))
width = 0.32

for row_idx, run in enumerate(["BASELINE", "NEW"]):
    for col_idx, kd_state in enumerate(["non-KD", "KD"]):
        ax = axes[row_idx, col_idx]
        sub = table5[(table5.run == run) & (table5.kd_state == kd_state)].set_index("eval_set").loc[groups]
        ax.bar([i - width / 2 for i in xg], sub.plain_acc_pct, width=width, color=MUTED, label="plain RankExt", zorder=3)
        ax.bar([i + width / 2 for i in xg], sub.orth_acc_pct, width=width, color=VIOLET, label="RankExt+FactorOrth", zorder=3)
        for i, (pv, ov) in enumerate(zip(sub.plain_acc_pct, sub.orth_acc_pct)):
            ax.text(i - width / 2, pv + 1.5, f"{pv:.1f}", ha="center", fontsize=8, color=INK2)
            ax.text(i + width / 2, ov + 1.5, f"{ov:.1f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(list(xg))
        ax.set_xticklabels(["first_step", "later_steps", "all_seen"], fontsize=9)
        ax.set_title(f"{run} — {kd_state}", fontsize=11, color=INK, loc="left")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
        if col_idx == 0:
            ax.set_ylabel("accuracy (%)")

axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Factor-orth flips from net-positive (BASELINE) to net-negative (NEW) for rank_extension,\n"
             "worst for the non-KD variant: first_step collapses to 0.1% in NEW",
             fontsize=11.5, color=INK2, y=1.02)
plt.tight_layout()
plt.savefig(OUT + "fig4_rankext_vs_rankext_orth.png", dpi=160, bbox_inches="tight")
plt.close()

print("Figures written to", OUT)
