#!/usr/bin/env python
# coding: utf-8
"""
Task 3 (decision doc, 2026-08-17): post-hoc regeneration of the simple_avg
convergence figure from the ALREADY-COMPLETED R3 run's saved CSVs -- no
training is re-run (per the user's explicit "do NOT launch training"
instruction). This script demonstrates, on real R3 data, the two visual fixes
now baked into vit_lora_cifar100_full5step_n5.py's refresh_live_convergence()
/ EpochValidationCallback for the NEXT real training run:

  (a) mark each CL step's selected (best-val-CE) epoch on the val-CE curve.
  (b) truncate plain simple_avg's curve at the epoch adaptive per-step early
      stop (patience=3, min_epoch=3, scoped to "simple_avg" ONLY -- see
      adaptive_early_stop_applies_to_method() in the main script) would have
      fired at, by REPLAYING the same running-best-epoch/patience logic the
      real EpochValidationCallback.on_epoch_end() now runs, against the real
      per-epoch val-CE sequence R3 already recorded. simple_avg_factor_orth
      is plotted in full (untruncated) alongside it, to show directly that
      the same replayed logic never trips patience for it (its own best
      epoch trends late: 2, 5, 7, 9, 6 across steps).

Source data (read-only): R3/results_featanchor_20260814_light/
clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260814_172709/
tables/{training_loss_history_by_epoch.csv, best_epoch_selected_by_method_step.csv}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import PchipInterpolator
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # normalized (no literal ".." segment) --
# avoids exceeding Windows' 260-char MAX_PATH once joined with this run's
# long timestamped directory name below.
R3_TABLES = os.path.normpath(os.path.join(
    REPO_ROOT, "R3", "results_featanchor_20260814_light",
    "clip_vit_lora_cifar100_full_comparison_with_orth_rankext_EPOCH3_MAIN_20260814_172709",
    "tables",
))
OUT_DIR = HERE

# Same constants as ADAPTIVE_PER_STEP_EARLY_STOP_* in the main script.
PATIENCE = 3
MIN_EPOCH = 3


def simulate_stop_epoch(val_ce_by_epoch):
    """Replays EpochValidationCallback.on_epoch_end()'s running-best-epoch +
    patience logic against a real per-epoch val-CE sequence (1-indexed dict
    epoch -> val_ce). Returns the LAST epoch that would actually run before
    adaptive early-stop fires (i.e. the truncation point), or the final
    configured epoch if it never fires."""
    best_val_ce = float("inf")
    best_epoch = None
    epochs_sorted = sorted(val_ce_by_epoch.keys())
    for epoch in epochs_sorted:
        val_ce = val_ce_by_epoch[epoch]
        if not np.isnan(val_ce) and val_ce < best_val_ce:
            best_val_ce = val_ce
            best_epoch = epoch
        if (
            best_epoch is not None
            and epoch >= MIN_EPOCH
            and (epoch - best_epoch) >= PATIENCE
        ):
            return epoch  # training stops AFTER this epoch
    return epochs_sorted[-1]


def plot_step_broken_series(ax, df, y_col, color, label, x_col, lw=1.8, ms=4.5, marker="o", linestyle="-"):
    first = True
    for _, g in df.groupby("step_id", sort=True):
        xs = g[x_col].to_numpy(dtype=float)
        ys = g[y_col].to_numpy(dtype=float)
        valid = ~np.isnan(ys)
        xs, ys = xs[valid], ys[valid]
        if len(xs) == 0:
            continue
        if len(xs) >= 3 and HAVE_SCIPY:
            xs_dense = np.linspace(xs.min(), xs.max(), 25)
            ax.plot(xs_dense, PchipInterpolator(xs, ys)(xs_dense), color=color, lw=lw,
                     linestyle=linestyle, label=(label if first else None))
        else:
            ax.plot(xs, ys, color=color, lw=lw, linestyle=linestyle, label=(label if first else None))
        ax.plot(xs, ys, marker=marker, ms=ms, lw=0, color=color)
        first = False


def build_frame(epoch_df, best_df, method, apply_early_stop):
    m = epoch_df[epoch_df["method"] == method].copy()
    m = m[["cl_step", "local_epoch", "train_ce_loss", "val_ce_loss"]].rename(
        columns={"cl_step": "step_id", "local_epoch": "epoch"}
    )
    if apply_early_stop:
        kept_rows = []
        for step_id, g in m.groupby("step_id"):
            val_ce_by_epoch = dict(zip(g["epoch"], g["val_ce_loss"]))
            stop_epoch = simulate_stop_epoch(val_ce_by_epoch)
            kept_rows.append(g[g["epoch"] <= stop_epoch])
        m = pd.concat(kept_rows, ignore_index=True)
    m = m.sort_values(["step_id", "epoch"]).reset_index(drop=True)
    m["_global_epoch"] = np.arange(1, len(m) + 1)
    best = best_df[best_df["method_name"] == method]
    return m, best


def render(ax, epoch_df, best_df, method, apply_early_stop, title):
    df, best = build_frame(epoch_df, best_df, method, apply_early_stop)
    plot_step_broken_series(ax, df, "train_ce_loss", "#1f77b4", "train CE", x_col="_global_epoch")
    plot_step_broken_series(ax, df, "val_ce_loss", "#d62728", "val CE", x_col="_global_epoch")

    first_marker = True
    for _, r in best.iterrows():
        match = df[(df["step_id"] == int(r["step_id"])) & (df["epoch"] == int(r["selected_epoch"]))]
        if len(match) == 0:
            continue
        mrow = match.iloc[0]
        ax.plot(
            mrow["_global_epoch"], mrow["val_ce_loss"],
            marker="*", ms=14, mec="black", mew=0.6, color="gold", zorder=5, linestyle="None",
            label=("selected (best) epoch" if first_marker else None),
        )
        first_marker = False

    step_sizes = df.groupby("step_id").size()
    boundary = 0
    for step_id in sorted(df["step_id"].unique())[:-1]:
        boundary += int(step_sizes.loc[step_id])
        ax.axvline(boundary + 0.5, color="gray", lw=0.7, ls=":", alpha=0.6)

    ax.set_xlabel("global epoch so far (dotted = CL step boundary)")
    ax.set_ylabel("CE loss")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)


def main():
    epoch_df = pd.read_csv(os.path.join(R3_TABLES, "training_loss_history_by_epoch.csv"))
    best_df = pd.read_csv(os.path.join(R3_TABLES, "best_epoch_selected_by_method_step.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    render(axes[0, 0], epoch_df, best_df, "simple_avg", apply_early_stop=False,
           title="simple_avg -- BEFORE (raw R3 curve, full 9 epochs/step, no marker)")
    render(axes[0, 1], epoch_df, best_df, "simple_avg", apply_early_stop=True,
           title="simple_avg -- AFTER (best-epoch marked + adaptive early-stop, patience=3)")
    render(axes[1, 0], epoch_df, best_df, "simple_avg_factor_orth", apply_early_stop=False,
           title="simple_avg_factor_orth -- BEFORE (raw R3 curve)")
    # NOTE: the deployed fix gates early-stop on method_name == "simple_avg"
    # explicitly (adaptive_early_stop_applies_to_method() in the main
    # script), so simple_avg_factor_orth's "AFTER" panel is the SAME raw
    # curve, not a patience replay. This is deliberate, not a shortcut: a
    # pure patience replay (see the printed simulated_stop_epoch below) would
    # have ALSO truncated its steps 1-2 (stop_epoch 5, 8) -- contradicting
    # the "never fires on simple_avg_factor_orth" assumption the now-removed
    # DEFAULT-OFF flag's original comment made, which only held for steps
    # 3-5 (its late-trending best epochs). The explicit method gate is what
    # actually guarantees "must NOT stop simple_avg_factor_orth early", not
    # patience dynamics alone.
    render(axes[1, 1], epoch_df, best_df, "simple_avg_factor_orth", apply_early_stop=False,
           title="simple_avg_factor_orth -- AFTER fix deployed (explicitly gated OFF -- unaffected)")

    fig.suptitle(
        "Task 3: simple_avg convergence fix, replayed on the completed R3 run's real per-epoch data\n"
        "(post-hoc demonstration -- no retraining; the actual code fix lives in "
        "refresh_live_convergence()/EpochValidationCallback for the next real run)",
        fontsize=11.5,
    )
    out_path = os.path.join(OUT_DIR, "simple_avg_convergence_fix_demo.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # Print the simulated stop epoch per step for both methods, for the report.
    for method in ["simple_avg", "simple_avg_factor_orth"]:
        m = epoch_df[epoch_df["method"] == method]
        print(f"\n{method}:")
        for step_id, g in m.groupby("cl_step"):
            val_ce_by_epoch = dict(zip(g["local_epoch"], g["val_ce_loss"]))
            stop_epoch = simulate_stop_epoch(val_ce_by_epoch)
            best_epoch = int(best_df[(best_df.method_name == method) & (best_df.step_id == step_id)]["selected_epoch"].iloc[0])
            print(f"  step {step_id}: best_epoch={best_epoch} | simulated_stop_epoch={stop_epoch} | configured=9")


if __name__ == "__main__":
    main()
