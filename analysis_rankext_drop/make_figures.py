import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- palette (dataviz skill reference palette, light mode) ----
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
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10.5,
})

NEW_DIR = "R3/results_4mod_20260715_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260715_144651/tables/"
PREV_DIR = "R3/results_EPOCH6_20260711_light/clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260711_214100/tables/"
OUT = "analysis_rankext_drop/"

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

# ============================================================
# Figure 1: accuracy delta bar chart (all_seen), all 8 methods
# ============================================================
new = pd.read_csv(NEW_DIR + "final_accuracy_selected_methods.csv")
prev = pd.read_csv(PREV_DIR + "final_accuracy_selected_methods.csv")
merged = new.merge(prev, on="method", suffixes=("_NEW", "_PREV"))
merged["delta_all_seen"] = merged["all_seen_NEW"] - merged["all_seen_PREV"]
merged["family"] = merged.method.apply(lambda m: "rank_extension" if m.startswith("rank_extension") else "simple_avg")
merged["label"] = merged.method.map(DISPLAY)
merged = merged.sort_values("delta_all_seen")

fig, ax = plt.subplots(figsize=(10.5, 5.2))
colors = [RED if v < 0 else BLUE for v in merged.delta_all_seen]
bars = ax.barh(merged.label, merged.delta_all_seen, color=colors, height=0.6, zorder=3)
ax.axvline(0, color=INK, linewidth=1)
for bar, v in zip(bars, merged.delta_all_seen):
    x = bar.get_width()
    ax.text(x + (1.2 if x >= 0 else -1.2), bar.get_y() + bar.get_height() / 2,
            f"{v:+.1f}", va="center", ha="left" if x >= 0 else "right",
            fontsize=9.5, color=INK)

# mark family boundary
ax.set_title("Change in all-seen accuracy: NEW (4mod+calib+headLRx10) vs PREV (20260711)",
             fontsize=11.5, color=INK, pad=14, loc="left")
ax.set_xlabel("Δ all_seen accuracy (percentage points)")
ax.set_xlim(min(merged.delta_all_seen.min() - 8, -10), max(merged.delta_all_seen.max() + 8, 10))
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="y", visible=False)

# annotate KD-in-rank_extension as the broken group
for lbl in ["RankExt+KD", "RankExt+FactorOrth+KD"]:
    ax.get_yticklabels()[list(merged.label).index(lbl)].set_color(RED)
    ax.get_yticklabels()[list(merged.label).index(lbl)].set_fontweight("bold")

plt.tight_layout()
plt.savefig(OUT + "fig1_accuracy_deltas.png", dpi=160)
plt.close()

# ============================================================
# Figure 2: rank_ext convergence overlay (val CE), NEW vs PREV
# ============================================================
new_conv = pd.read_csv(NEW_DIR + "all_methods_convergence_table.csv")
prev_conv = pd.read_csv(PREV_DIR + "all_methods_convergence_table.csv")

methods_to_plot = ["rank_extension_kd_only_T2", "rank_extension_orth_factor_lam_50_kd_T2"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

for ax, m in zip(axes, methods_to_plot):
    for df, run_label, style in [(new_conv, "NEW (9ep, 4mod)", "-"), (prev_conv, "PREV (6ep, 2mod)", "--")]:
        sub = df[df.method_name == m].copy()
        # build a continuous x-axis: step_id offset by cumulative epochs
        sub = sub.sort_values(["step_id", "epoch"])
        x = []
        offset = 0
        last_step = None
        step_boundaries = []
        for _, row in sub.iterrows():
            if last_step is not None and row.step_id != last_step:
                offset = len(x)
                step_boundaries.append(offset)
            x.append(offset + row.epoch)
            last_step = row.step_id
        color = BLUE if "NEW" in run_label else ORANGE
        ax.plot(range(len(sub)), sub.val_ce_loss.values, style, color=color, linewidth=2,
                label=run_label, zorder=3)
    # step boundaries for NEW (denser) as light vlines
    sub_new = new_conv[new_conv.method_name == m].sort_values(["step_id", "epoch"])
    boundaries = sub_new.groupby("step_id").size().cumsum().values[:-1]
    for b in boundaries:
        ax.axvline(b - 0.5, color=GRID, linewidth=1, zorder=1)
    ax.set_title(DISPLAY[m], fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel("epoch index (concatenated across CL steps 1→5)")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("validation CE loss")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle("RankExt+KD variants: per-step training/val CE looks healthy in BOTH runs\n"
             "(the accuracy collapse happens after training, not during it)",
             fontsize=11.5, color=INK2, y=1.04)
plt.tight_layout()
plt.savefig(OUT + "fig2_convergence_overlay.png", dpi=160, bbox_inches="tight")
plt.close()

# ============================================================
# Figure 3: per-group accuracy pattern (first_step vs later_steps)
# ============================================================
rankext_methods = ["rank_extension", "rank_extension_kd_only_T2",
                   "rank_extension_orth_factor_lam_50", "rank_extension_orth_factor_lam_50_kd_T2"]

fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
groups = ["first_step", "later_steps"]
x = range(len(groups))
width = 0.32

for ax, m in zip(axes, rankext_methods):
    prev_row = merged[merged.method == m].iloc[0]
    new_vals = [new[new.method == m][g].iloc[0] for g in groups]
    prev_vals = [prev[prev.method == m][g].iloc[0] for g in groups]
    ax.bar([i - width/2 for i in x], prev_vals, width=width, color=ORANGE, label="PREV", zorder=3)
    ax.bar([i + width/2 for i in x], new_vals, width=width, color=BLUE, label="NEW", zorder=3)
    for i, (pv, nv) in enumerate(zip(prev_vals, new_vals)):
        ax.text(i - width/2, pv + 1.5, f"{pv:.0f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + width/2, nv + 1.5, f"{nv:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["first_step\n(step 1)", "later_steps\n(avg steps 2-5)"], fontsize=9)
    ax.set_title(DISPLAY[m], fontsize=11, color=INK)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("accuracy (%)")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
fig.suptitle("first_step rises while later_steps collapses for the two KD rank_extension variants\n"
             "(non-KD variants show no such split)",
             fontsize=11.5, color=INK2, y=1.05)
plt.tight_layout()
plt.savefig(OUT + "fig3_per_group_pattern.png", dpi=160, bbox_inches="tight")
plt.close()

print("Figures written to", OUT)
