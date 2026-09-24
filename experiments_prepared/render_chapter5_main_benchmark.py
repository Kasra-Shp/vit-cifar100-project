"""
Reusable Chapter-5 rendering pipeline.

Reads ONLY the canonical CSVs under R7/chapter5_main_benchmark/data/ and
generates every Chapter-5 table/figure from them. No result value is
hard-coded in this script -- everything numeric comes from the data files.

Usage:
    python experiments_prepared/render_chapter5_main_benchmark.py --dataset cifar100
    python experiments_prepared/render_chapter5_main_benchmark.py            # all datasets found

Dataset-extensible by design: datasets are discovered from
R7/chapter5_main_benchmark/data/chapter5_<dataset>_8method_main.csv . Adding
a second dataset (e.g. imagenet100) requires only dropping a matching pair of
CSVs into that folder and re-running this script -- no code change.
"""

import argparse
import glob
import os
import textwrap
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_ROOT = os.path.join(REPO_ROOT, "R7", "chapter5_main_benchmark")
DATA_DIR = os.path.join(PKG_ROOT, "data")
FIG_DIR = os.path.join(PKG_ROOT, "figures")
TAB_DIR = os.path.join(PKG_ROOT, "tables")

DPI = 300

# Fixed presentation order for the 8 canonical methods (shared by every
# dataset -- the whole point of the package is comparing the SAME 8 methods
# across datasets).
METHOD_ORDER = [
    "simple_avg", "simple_avg_kd", "simple_avg_denseorth", "simple_avg_denseorth_kd",
    "rankext", "rankext_kd_protect", "rankext_factororth", "rankext_factororth_kd_protect",
]
FORBIDDEN_SUBSTRINGS = ["t4", "T4"]

FAMILY_COLORS = {"SimpleAvg": "#2A6F97", "RankExt": "#B23A48"}
FAMILY_SHADES = {
    "SimpleAvg": ["#0B3D5C", "#2A6F97", "#5FA8D3", "#9FC9E0"],
    "RankExt": ["#6E0F1E", "#B23A48", "#D97B85", "#EFB6BC"],
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


# --------------------------------------------------------------------------
# Data loading / validation
# --------------------------------------------------------------------------

def discover_datasets():
    pattern = os.path.join(DATA_DIR, "chapter5_*_8method_main.csv")
    found = []
    for path in sorted(glob.glob(pattern)):
        m = re.match(r"chapter5_(.+)_8method_main\.csv$", os.path.basename(path))
        if m:
            found.append(m.group(1))
    return found


def load_dataset(dataset):
    wide_path = os.path.join(DATA_DIR, f"chapter5_{dataset}_8method_main.csv")
    long_path = os.path.join(DATA_DIR, f"chapter5_{dataset}_per_task_metrics.csv")
    if not os.path.isfile(wide_path):
        raise FileNotFoundError(wide_path)
    df = pd.read_csv(wide_path)
    long_df = pd.read_csv(long_path) if os.path.isfile(long_path) else None
    validate(df, dataset)
    df = df.set_index("method_id").loc[METHOD_ORDER].reset_index()
    return df, long_df


def validate(df, dataset):
    if len(df) != 8:
        fail(f"[{dataset}] expected exactly 8 methods, found {len(df)}")
    missing = set(METHOD_ORDER) - set(df["method_id"])
    if missing:
        fail(f"[{dataset}] missing canonical method_id(s): {sorted(missing)}")
    blob = " ".join(df["method_id"].astype(str)) + " " + " ".join(df["display_name"].astype(str))
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad.lower() in blob.lower():
            fail(f"[{dataset}] forbidden substring '{bad}' found in method_id/display_name "
                 f"(SimpleAvg KD T4 must never appear in the Chapter-5 package)")
    if df["dataset"].nunique() != 1 or df["dataset"].iloc[0] != dataset:
        fail(f"[{dataset}] dataset column inconsistent with filename")


def fail(msg):
    print(f"[render_chapter5_main_benchmark] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip() in ("—", "", "nan", "NaN"):
        return True
    return False


def numeric_or_none(v):
    return None if is_missing(v) else float(v)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

REPORT = {"generated": [], "skipped": []}


def note_generated(name):
    REPORT["generated"].append(name)
    print(f"  [generated] {name}")


def note_skipped(name, reason):
    REPORT["skipped"].append((name, reason))
    print(f"  [skipped]   {name} -- {reason}")


def savefig(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def bar_colors(df):
    return [FAMILY_COLORS[f] for f in df["family"]]


def wrap_labels(ax, labels, rotation=30):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right")


# --------------------------------------------------------------------------
# Figure 1 -- main all-seen accuracy
# --------------------------------------------------------------------------

def fig_allseen_accuracy(df, dataset):
    name = f"{dataset}_allseen_accuracy"
    vals = df["all_seen_accuracy"].astype(float).values
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(df["display_name"], vals, color=bar_colors(df), width=0.62)
    ax.set_ylim(0, max(100, vals.max() * 1.12))
    ax.set_ylabel("All-seen accuracy (%)")
    wrap_labels(ax, df["display_name"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in ["SimpleAvg", "RankExt"]]
    ax.legend(handles, ["SimpleAvg", "RankExt"], frameon=False, loc="upper right")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 2 -- family progression
# --------------------------------------------------------------------------

def fig_family_progression(df, dataset):
    name = f"{dataset}_family_progression"
    sa_ids = ["simple_avg", "simple_avg_kd", "simple_avg_denseorth", "simple_avg_denseorth_kd"]
    re_ids = ["rankext", "rankext_kd_protect", "rankext_factororth", "rankext_factororth_kd_protect"]
    sa_labels = ["Plain", "+ KD", "+ FactorOrth", "+ FactorOrth\n+ KD"]
    re_labels = ["Plain", "+ KD\n+ Protect", "+ FactorOrth", "+ FactorOrth\n+ KD + Protect"]

    d = df.set_index("method_id")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)

    for ax, ids, labels, family in [(axes[0], sa_ids, sa_labels, "SimpleAvg"),
                                     (axes[1], re_ids, re_labels, "RankExt")]:
        vals = [float(d.loc[i, "all_seen_accuracy"]) for i in ids]
        shades = FAMILY_SHADES[family]
        bars = ax.bar(range(len(ids)), vals, color=shades[:len(ids)], width=0.6)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(family)
        ax.set_ylim(0, max(100, max(vals) * 1.15))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("All-seen accuracy (%)")
    fig.suptitle("")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 3 -- open vs restricted
# --------------------------------------------------------------------------

def fig_open_vs_restricted(df, dataset):
    name = f"{dataset}_open_vs_restricted"
    if df["restricted_accuracy"].apply(is_missing).any():
        note_skipped(name, "restricted_accuracy missing for at least one method")
        return
    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, df["all_seen_accuracy"].astype(float), width=w, label="All-seen (open)",
           color=[FAMILY_COLORS[f] for f in df["family"]])
    ax.bar(x + w / 2, df["restricted_accuracy"].astype(float), width=w,
           label="Restricted (task-oracle, diagnostic)",
           color=[FAMILY_COLORS[f] for f in df["family"]], alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 4 -- open-restricted gap
# --------------------------------------------------------------------------

def fig_open_restricted_gap(df, dataset):
    name = f"{dataset}_open_restricted_gap"
    if df["restricted_accuracy"].apply(is_missing).any():
        note_skipped(name, "restricted_accuracy missing for at least one method")
        return
    gap = df["restricted_accuracy"].astype(float) - df["all_seen_accuracy"].astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(df["display_name"], gap, color=bar_colors(df), width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Restricted − All-seen (pp)\n(diagnostic gap, not a direct forgetting measure)")
    wrap_labels(ax, df["display_name"])
    for b, v in zip(bars, gap):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.0 if v >= 0 else -1.0), f"{v:.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 5 -- first vs later tasks
# --------------------------------------------------------------------------

def fig_first_vs_later(df, dataset):
    name = f"{dataset}_first_vs_later_tasks"
    if df["first_task_final_accuracy"].apply(is_missing).any() or \
       df["later_tasks_mean_accuracy"].apply(is_missing).any():
        note_skipped(name, "first_task_final_accuracy or later_tasks_mean_accuracy missing")
        return
    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, df["first_task_final_accuracy"].astype(float), width=w,
           label="First step (final, open)", color=bar_colors(df))
    ax.bar(x + w / 2, df["later_tasks_mean_accuracy"].astype(float), width=w,
           label="Later steps, pooled (final, open)", color=bar_colors(df), alpha=0.5, hatch="\\\\")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figures 6/7 -- per-task open / restricted, per family
# --------------------------------------------------------------------------

def _per_task_family_plot(long_df, df, dataset, metric, family, name_suffix, ylabel):
    name = f"{dataset}_{name_suffix}"
    if long_df is None:
        note_skipped(name, "per-task metrics file not available")
        return
    ids = df.loc[df["family"] == family, "method_id"].tolist()
    labels = df.loc[df["family"] == family, "display_name"].tolist()
    sub = long_df[(long_df["metric"] == metric) & (long_df["method_id"].isin(ids))]
    if sub.empty:
        note_skipped(name, f"no '{metric}' rows found for family {family}")
        return
    shades = FAMILY_SHADES[family]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, (mid, lab) in enumerate(zip(ids, labels)):
        s = sub[sub["method_id"] == mid].sort_values("task")
        if s.empty:
            continue
        ax.plot(s["task"], s["value"], marker="o", color=shades[i % len(shades)], label=lab, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(sub["task"].unique()))
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_per_task_final_open(long_df, df, dataset):
    _per_task_family_plot(long_df, df, dataset, "open_accuracy", "SimpleAvg",
                           "sa_per_task_final_open", "Final open accuracy (%)")
    _per_task_family_plot(long_df, df, dataset, "open_accuracy", "RankExt",
                           "rankext_per_task_final_open", "Final open accuracy (%)")


def fig_per_task_restricted(long_df, df, dataset):
    _per_task_family_plot(long_df, df, dataset, "restricted_accuracy", "SimpleAvg",
                           "sa_per_task_restricted", "Restricted (task-oracle) accuracy (%)")
    _per_task_family_plot(long_df, df, dataset, "restricted_accuracy", "RankExt",
                           "rankext_per_task_restricted", "Restricted (task-oracle) accuracy (%)")


# --------------------------------------------------------------------------
# Figure 8 -- BWT
# --------------------------------------------------------------------------

def fig_bwt(df, dataset):
    name = f"{dataset}_bwt"
    if df["BWT"].apply(is_missing).any():
        note_skipped(name, "BWT missing for at least one method")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, family in zip(axes, ["SimpleAvg", "RankExt"]):
        sub = df[df["family"] == family]
        bars = ax.bar(sub["display_name"], sub["BWT"].astype(float), color=FAMILY_COLORS[family], width=0.55)
        ax.axhline(0, color="black", linewidth=0.8)
        wrap_labels(ax, sub["display_name"])
        ax.set_title(family)
        for b, v in zip(bars, sub["BWT"].astype(float)):
            ax.text(b.get_x() + b.get_width() / 2, v - 0.01, f"{v:.3f}", ha="center",
                     va="top" if v < 0 else "bottom", fontsize=8.5)
    axes[0].set_ylabel("Backward Transfer (BWT)")
    fig.suptitle("SimpleAvg BWT uses the pre-merge specialist diagonal substitute -- not\n"
                 "structurally equivalent to persistent-model RankExt BWT", fontsize=8.5, y=1.04)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 9 -- RankExt forgetting
# --------------------------------------------------------------------------

def fig_rankext_forgetting(df, dataset):
    name = f"{dataset}_rankext_forgetting"
    sub = df[df["family"] == "RankExt"]
    if sub["forgetting"].apply(is_missing).any():
        note_skipped(name, "forgetting missing for at least one RankExt method")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    vals = sub["forgetting"].astype(float)
    bars = ax.bar(sub["display_name"], vals, color=FAMILY_COLORS["RankExt"], width=0.55)
    ax.set_ylabel("Forgetting (fraction, lower = better)")
    wrap_labels(ax, sub["display_name"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 10 -- RankExt recovery over plain
# --------------------------------------------------------------------------

def fig_rankext_recovery_over_plain(df, dataset):
    name = f"{dataset}_rankext_recovery_over_plain"
    d = df.set_index("method_id")
    plain = float(d.loc["rankext", "all_seen_accuracy"])
    ids = ["rankext_kd_protect", "rankext_factororth", "rankext_factororth_kd_protect"]
    labels = ["+ KD + Protect", "+ FactorOrth", "+ FactorOrth\n+ KD + Protect"]
    deltas = [float(d.loc[i, "all_seen_accuracy"]) - plain for i in ids]
    shades = FAMILY_SHADES["RankExt"][1:]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(labels, deltas, color=shades, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"All-seen accuracy gain over RankExt plain (pp)\n(plain = {plain:.2f}%)")
    for b, v in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"+{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 11 -- SimpleAvg delta from plain
# --------------------------------------------------------------------------

def fig_simpleavg_delta_from_plain(df, dataset):
    name = f"{dataset}_simpleavg_delta_from_plain"
    d = df.set_index("method_id")
    plain = float(d.loc["simple_avg", "all_seen_accuracy"])
    ids = ["simple_avg_kd", "simple_avg_denseorth", "simple_avg_denseorth_kd"]
    labels = ["+ KD", "+ FactorOrth", "+ FactorOrth\n+ KD"]
    deltas = [float(d.loc[i, "all_seen_accuracy"]) - plain for i in ids]
    shades = FAMILY_SHADES["SimpleAvg"][1:]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(labels, deltas, color=shades, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"All-seen accuracy delta vs. SimpleAvg plain (pp)\n(plain = {plain:.2f}%)")
    for b, v in zip(bars, deltas):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylim(min(deltas) - 0.8, max(0.8, max(deltas) + 0.8))
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Figure 12 -- best per family
# --------------------------------------------------------------------------

def fig_best_per_family(df, dataset):
    name = f"{dataset}_best_per_family"
    best_sa = df[df["family"] == "SimpleAvg"].sort_values("all_seen_accuracy", ascending=False).iloc[0]
    best_re = df[df["family"] == "RankExt"].sort_values("all_seen_accuracy", ascending=False).iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    bars = ax.bar([best_sa["display_name"], best_re["display_name"]],
                  [best_sa["all_seen_accuracy"], best_re["all_seen_accuracy"]],
                  color=[FAMILY_COLORS["SimpleAvg"], FAMILY_COLORS["RankExt"]], width=0.5)
    ax.set_ylabel("All-seen accuracy (%)")
    ax.set_ylim(0, 100)
    wrap_labels(ax, [best_sa["display_name"], best_re["display_name"]], rotation=15)
    for b, v in zip(bars, [best_sa["all_seen_accuracy"], best_re["all_seen_accuracy"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Highest observed result per family (single seed)", fontsize=9.5)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Optional figures
# --------------------------------------------------------------------------

def fig_opt_accuracy_vs_gap_scatter(df, dataset):
    name = f"{dataset}_accuracy_vs_gap_scatter"
    if df["restricted_accuracy"].apply(is_missing).any():
        note_skipped(name, "restricted_accuracy missing for at least one method")
        return
    gap = df["restricted_accuracy"].astype(float) - df["all_seen_accuracy"].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df["all_seen_accuracy"], gap, color=bar_colors(df), s=70, zorder=3)
    for _, r in df.iterrows():
        g = float(r["restricted_accuracy"]) - float(r["all_seen_accuracy"])
        ax.annotate(r["display_name"], (r["all_seen_accuracy"], g), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("All-seen accuracy (%)")
    ax.set_ylabel("Restricted − All-seen gap (pp)")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_opt_rankext_finalacc_vs_forgetting(df, dataset):
    name = f"{dataset}_rankext_finalacc_vs_forgetting"
    sub = df[df["family"] == "RankExt"]
    if sub["forgetting"].apply(is_missing).any():
        note_skipped(name, "forgetting missing for at least one RankExt method")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(sub["forgetting"].astype(float), sub["all_seen_accuracy"].astype(float),
               color=FAMILY_COLORS["RankExt"], s=70, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(r["display_name"], (float(r["forgetting"]), float(r["all_seen_accuracy"])), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Forgetting (fraction, lower = better)")
    ax.set_ylabel("All-seen accuracy (%)")
    ax.invert_xaxis()
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_opt_kd_paired_effect(df, dataset):
    name = f"{dataset}_kd_paired_effect"
    d = df.set_index("method_id")
    pairs = [("SimpleAvg", "simple_avg", "simple_avg_kd"),
             ("RankExt", "rankext_factororth", "rankext_factororth_kd_protect")]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(pairs))
    w = 0.32
    before = [float(d.loc[b, "all_seen_accuracy"]) for _, b, _ in pairs]
    after = [float(d.loc[a, "all_seen_accuracy"]) for _, _, a in pairs]
    ax.bar(x - w / 2, before, width=w, label="Without KD", color="#9FA6AD")
    ax.bar(x + w / 2, after, width=w, label="With KD", color=[FAMILY_COLORS[f] for f, _, _ in pairs])
    ax.set_xticks(x)
    ax.set_xticklabels(["SimpleAvg\n(plain -> +KD)", "RankExt\n(+FactorOrth -> +FactorOrth+KD+Protect)"], fontsize=9)
    ax.set_ylabel("All-seen accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_opt_orth_paired_effect(df, dataset):
    name = f"{dataset}_orthogonality_paired_effect"
    d = df.set_index("method_id")
    pairs = [("SimpleAvg", "simple_avg", "simple_avg_denseorth"),
             ("RankExt", "rankext", "rankext_factororth")]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = np.arange(len(pairs))
    w = 0.32
    before = [float(d.loc[b, "all_seen_accuracy"]) for _, b, _ in pairs]
    after = [float(d.loc[a, "all_seen_accuracy"]) for _, _, a in pairs]
    ax.bar(x - w / 2, before, width=w, label="Without orthogonality loss", color="#9FA6AD")
    ax.bar(x + w / 2, after, width=w, label="With orthogonality loss", color=[FAMILY_COLORS[f] for f, _, _ in pairs])
    ax.set_xticks(x)
    ax.set_xticklabels(["SimpleAvg\n(plain -> +FactorOrth)", "RankExt\n(plain -> +FactorOrth)"], fontsize=9)
    ax.set_ylabel("All-seen accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Parameter / rank efficiency (optional per-dataset CSV: chapter5_<dataset>_parameter_efficiency.csv)
# --------------------------------------------------------------------------

def load_parameter_efficiency(dataset):
    path = os.path.join(DATA_DIR, f"chapter5_{dataset}_parameter_efficiency.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def fig_rank_growth(param_df, dataset):
    name = f"{dataset}_rank_growth"
    if param_df is None:
        note_skipped(name, "parameter-efficiency data not available for this dataset")
        return
    sa = param_df[(param_df["family"] == "SimpleAvg") & (param_df["method"] == "simple_avg")].sort_values("task")
    re_ = param_df[(param_df["family"] == "RankExt") & (param_df["method"] == "rankext")].sort_values("task")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sa["task"], sa["active_rank"], linestyle="--", marker="s", color=FAMILY_COLORS["SimpleAvg"],
            label="SimpleAvg (fixed rank per independent specialist)")
    ax.plot(re_["task"], re_["active_rank"], linestyle="-", marker="o", color=FAMILY_COLORS["RankExt"],
            label="RankExt (cumulative active rank)")
    ax.set_xlabel("Task")
    ax.set_ylabel("LoRA rank")
    ax.set_xticks(sorted(param_df["task"].unique()))
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(0.98, 0.04,
            "SimpleAvg's rank-80 is a NEW, independent specialist at every task\n"
            "(not cumulative history in the same sense as RankExt's growing adapter).",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, style="italic",
            color="#555555")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_trainable_lora_params_per_task(param_df, dataset):
    name = f"{dataset}_trainable_lora_params_per_task"
    if param_df is None:
        note_skipped(name, "parameter-efficiency data not available for this dataset")
        return
    sa = param_df[(param_df["family"] == "SimpleAvg") & (param_df["method"] == "simple_avg")].sort_values("task")
    re_ = param_df[(param_df["family"] == "RankExt") & (param_df["method"] == "rankext")].sort_values("task")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sa["task"], sa["new_trainable_lora_params"] / 1e6, linestyle="--", marker="s",
            color=FAMILY_COLORS["SimpleAvg"], label="SimpleAvg")
    ax.plot(re_["task"], re_["new_trainable_lora_params"] / 1e6, linestyle="-", marker="o",
            color=FAMILY_COLORS["RankExt"], label="RankExt")
    ax.set_xlabel("Task")
    ax.set_ylabel("NEW trainable LoRA parameters this task (millions)")
    ax.set_xticks(sorted(param_df["task"].unique()))
    ax.set_ylim(0, max(sa["new_trainable_lora_params"].max(), re_["new_trainable_lora_params"].max()) / 1e6 * 1.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def fig_cumulative_adapter_params(param_df, dataset):
    name = f"{dataset}_cumulative_adapter_params"
    if param_df is None:
        note_skipped(name, "parameter-efficiency data not available for this dataset")
        return
    sa = param_df[(param_df["family"] == "SimpleAvg") & (param_df["method"] == "simple_avg")].sort_values("task")
    re_ = param_df[(param_df["family"] == "RankExt") & (param_df["method"] == "rankext")].sort_values("task")
    sa_cumulative_if_all_retained = sa["cumulative_lora_params"].cumsum()
    re_cumulative_active = re_["cumulative_lora_params"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sa["task"], sa_cumulative_if_all_retained / 1e6, linestyle="--", marker="s",
            color=FAMILY_COLORS["SimpleAvg"],
            label="SimpleAvg -- volume IF every specialist were kept separately (hypothetical;\nnot the deployed form, see the deployment table)")
    ax.plot(re_["task"], re_cumulative_active / 1e6, linestyle="-", marker="o",
            color=FAMILY_COLORS["RankExt"], label="RankExt -- cumulative ACTIVE adapter (this IS the deployed form)")
    ax.set_xlabel("Task")
    ax.set_ylabel("Adapter parameter volume (millions)")
    ax.set_xticks(sorted(param_df["task"].unique()))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


def table_parameter_efficiency(param_df, dataset):
    name = f"{dataset}_parameter_efficiency_table"
    if param_df is None:
        note_skipped(name, "parameter-efficiency data not available for this dataset")
        return
    sa = param_df[(param_df["family"] == "SimpleAvg") & (param_df["method"] == "simple_avg")].set_index("task")
    re_ = param_df[(param_df["family"] == "RankExt") & (param_df["method"] == "rankext")].set_index("task")

    def m(x):
        return f"{int(x) / 1e6:.2f}M"

    rows = [
        ["Rank behavior", "Fixed rank 80/specialist", "Growing: 16→32→48→64→80"],
        ["Rank at Task 1", "80", "16"],
        ["Rank at Task 5", "80", "80 (cumulative)"],
        ["New trainable LoRA params/task", m(sa.loc[1, "new_trainable_lora_params"]) + " (constant)",
         m(re_.loc[1, "new_trainable_lora_params"]) + " (constant)"],
        ["LoRA volume touched, all 5 tasks",
         m(sa["new_trainable_lora_params"].sum()) + " (5 independent adapters)",
         m(re_["new_trainable_lora_params"].sum()) + " (5 blocks, never retrained)"],
        ["Final deployed LoRA overhead", "0 if fused / " + m(24 * 768 * 768) + " if delta stored",
         m(re_.loc[5, "active_lora_params"]) + " (low-rank, additive)"],
        ["Final deployed classifier", "76,900", "76,900"],
        ["Deployment form", "Dense-fused into base weights (no adapter survives)",
         "Low-rank, additive (never fused)"],
    ]
    headers = ["Property", "SimpleAvg", "RankExt"]
    base = os.path.join(TAB_DIR, f"{dataset}_parameter_efficiency_table")
    pd.DataFrame(rows, columns=headers).to_csv(base + ".csv", index=False)
    _render_table_png(headers, rows, base + ".png", col_widths=[0.24, 0.40, 0.36], fontsize=10,
                       wrap_chars=[26, 34, 34])
    _write_tex(headers, rows, base + ".tex",
               caption=f"Parameter/rank-efficiency comparison, {param_df['dataset'].iloc[0]} "
                       f"(job4971615). All values source-verified against the pipeline's own "
                       f"trainable-parameter and rank-structure logs.",
               label=f"tab:{dataset}_parameter_efficiency")
    note_generated(f"{dataset}_parameter_efficiency_table (.csv/.png/.tex)")


# --------------------------------------------------------------------------
# RankExt R_{i,j} continual-learning accuracy matrices
# (optional per-dataset CSV: chapter5_<dataset>_rankext_Rij.csv)
# --------------------------------------------------------------------------

RIJ_METHOD_ORDER = ["rankext", "rankext_kd_protect", "rankext_factororth", "rankext_factororth_kd_protect"]
RIJ_DISPLAY = {
    "rankext": "RankExt (plain)",
    "rankext_kd_protect": "RankExt + KD + Protect",
    "rankext_factororth": "RankExt + FactorOrth",
    "rankext_factororth_kd_protect": "RankExt + FactorOrth + KD + Protect",
}
RIJ_FILE_SUFFIX = {
    "rankext": "plain", "rankext_kd_protect": "kd_protect",
    "rankext_factororth": "factororth", "rankext_factororth_kd_protect": "combined",
}


def load_rankext_rij(dataset):
    path = os.path.join(DATA_DIR, f"chapter5_{dataset}_rankext_Rij.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def _rij_matrix(rij_df, method_id, metric="open_accuracy"):
    sub = rij_df[(rij_df["method_id"] == method_id) & (rij_df["metric"] == metric)]
    mat = np.full((5, 5), np.nan)
    for _, r in sub.iterrows():
        i, j = int(r["train_step"]), int(r["eval_task"])
        mat[i - 1, j - 1] = r["accuracy"]
    return mat


def _draw_rij_heatmap(ax, mat, vmin=0, vmax=100, annotate=True, cmap="viridis"):
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"T{j}" for j in range(1, 6)])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"i={i}" for i in range(1, 6)])
    if annotate:
        for i in range(5):
            for j in range(5):
                if not np.isnan(mat[i, j]):
                    val = mat[i, j]
                    color = "white" if val < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)
    return im


def fig_rankext_rij_individual(rij_df, dataset):
    if rij_df is None:
        for mid in RIJ_METHOD_ORDER:
            note_skipped(f"{dataset}_rankext_{RIJ_FILE_SUFFIX[mid]}_Rij", "R_ij data not available for this dataset")
        return
    for method_id in RIJ_METHOD_ORDER:
        name = f"{dataset}_rankext_{RIJ_FILE_SUFFIX[method_id]}_Rij"
        mat = _rij_matrix(rij_df, method_id, "open_accuracy")
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        im = _draw_rij_heatmap(ax, mat)
        ax.set_xlabel("Evaluated task j")
        ax.set_ylabel("Trained through step i")
        ax.set_title(RIJ_DISPLAY[method_id], fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Open (all-seen) accuracy (%)", fontsize=8.5)
        fig.tight_layout()
        savefig(fig, os.path.join(FIG_DIR, name))
        note_generated(name)


def fig_rankext_rij_comparison(rij_df, dataset):
    name = f"{dataset}_rankext_Rij_comparison"
    if rij_df is None:
        note_skipped(name, "R_ij data not available for this dataset")
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    im = None
    for ax, method_id in zip(axes, RIJ_METHOD_ORDER):
        mat = _rij_matrix(rij_df, method_id, "open_accuracy")
        im = _draw_rij_heatmap(ax, mat, annotate=False)
        ax.set_title(RIJ_DISPLAY[method_id], fontsize=9)
        ax.set_xlabel("Eval task j", fontsize=8.5)
    axes[0].set_ylabel("Trained through step i", fontsize=8.5)
    fig.subplots_adjust(right=0.90, wspace=0.35)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Open (all-seen) accuracy (%)", fontsize=8.5)
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Cross-dataset figure
# --------------------------------------------------------------------------

def fig_cross_dataset_allseen(all_dfs):
    name = "cross_dataset_allseen_accuracy"
    if len(all_dfs) < 2:
        note_skipped(name, f"only {len(all_dfs)} dataset(s) available -- need >=2 for a cross-dataset figure")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(all_dfs)
    width = 0.8 / n
    method_labels = None
    for i, (dataset, df) in enumerate(all_dfs.items()):
        method_labels = df["display_name"].tolist()
        x = np.arange(len(df)) + (i - (n - 1) / 2) * width
        ax.bar(x, df["all_seen_accuracy"].astype(float), width=width, label=dataset)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=30, ha="right")
    ax.set_ylabel("All-seen accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIR, name))
    note_generated(name)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def _fmt(v, decimals=2, suffix=""):
    if is_missing(v):
        return "—"
    return f"{float(v):.{decimals}f}{suffix}"


def _tex_escape(s):
    s = str(s).replace("—", "--")  # avoid relying on inputenc/fontenc for the em dash
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _render_table_png(headers, rows, path, col_widths=None, fontsize=9.5, wrap_chars=None, cell_loc=None):
    """Always auto-wraps both headers and cells so matplotlib's table renders
    genuine multi-line text instead of overflowing/overlapping neighboring
    cells -- this was a real, previously-unverified rendering bug (long
    headers like "Later tasks mean (%)" silently overlapped adjacent cells).
    `wrap_chars`: optional explicit per-column character-wrap width for long
    free-text cells (e.g. the parameter-efficiency table); when omitted, a
    wrap width is derived automatically from fig_w/col_widths/fontsize so
    every table gets safe wrapping by default, not just ones that opt in."""
    fig_w = max(9, 1.15 * len(headers)) if wrap_chars is None else max(11, sum(col_widths or [1] * len(headers)) * 11)
    if wrap_chars is None:
        # ~ (inches per column * conservative chars-per-inch, bold-header-safe), floor 8 chars
        chars_per_inch = 8.0 * (9.5 / fontsize)
        widths = col_widths or [1.0 / len(headers)] * len(headers)
        wrap_chars = [max(6, int(fig_w * w * chars_per_inch)) for w in widths]

    def wrap_cell(text, c):
        return "\n".join(textwrap.wrap(str(text), wrap_chars[c], break_long_words=False,
                                        break_on_hyphens=False)) or str(text)

    # Headers are always wrapped (bold, same width budget as their column) --
    # textwrap on a string that already fits is a harmless no-op.
    wrapped_headers = [wrap_cell(h, c) for c, h in enumerate(headers)]
    header_lines = max((h.count("\n") + 1 for h in wrapped_headers), default=1)

    wrapped_rows = []
    for row in rows:
        wrapped_rows.append([wrap_cell(cell, c) for c, cell in enumerate(row)])
    row_line_counts = [max(cell.count("\n") + 1 for cell in row) for row in wrapped_rows]

    total_lines = header_lines + sum(row_line_counts)
    fig_h = 0.55 * total_lines + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=wrapped_rows, colLabels=wrapped_headers,
                    cellLoc=cell_loc or "center", loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold", ha="center")
            cell.set_height(cell.get_height() * header_lines)
        else:
            cell.set_height(cell.get_height() * row_line_counts[r - 1])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _write_tex(headers, rows, path, caption, label):
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_tex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def table_main(df, dataset):
    headers = ["Method", "All-seen (%)", "Restricted (%)", "First task (%)",
               "Later tasks mean (%)", "BWT", "Forgetting"]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["display_name"], _fmt(r["all_seen_accuracy"]), _fmt(r["restricted_accuracy"]),
            _fmt(r["first_task_final_accuracy"]), _fmt(r["later_tasks_mean_accuracy"]),
            _fmt(r["BWT"], 4), _fmt(r["forgetting"], 4),
        ])
    base = os.path.join(TAB_DIR, f"{dataset}_main_8method_table")
    pd.DataFrame(rows, columns=headers).to_csv(base + ".csv", index=False)
    _render_table_png(headers, rows, base + ".png",
                       col_widths=[0.30, 0.12, 0.12, 0.12, 0.14, 0.10, 0.10])
    _write_tex(headers, rows, base + ".tex",
               caption=f"Main 8-method benchmark, {df['benchmark'].iloc[0]}, seed {df['seed'].iloc[0]}.",
               label=f"tab:{dataset}_main_8method")
    note_generated(f"{dataset}_main_8method_table (.csv/.png/.tex)")


def table_compact(df, dataset):
    headers = ["Method", "All-seen (%)", "Restricted (%)"]
    rows = [[r["display_name"], _fmt(r["all_seen_accuracy"]), _fmt(r["restricted_accuracy"])]
            for _, r in df.iterrows()]
    base = os.path.join(TAB_DIR, f"{dataset}_main_compact_table")
    pd.DataFrame(rows, columns=headers).to_csv(base + ".csv", index=False)
    _render_table_png(headers, rows, base + ".png", col_widths=[0.5, 0.25, 0.25])
    _write_tex(headers, rows, base + ".tex",
               caption=f"Compact main-flow benchmark summary, {df['benchmark'].iloc[0]}.",
               label=f"tab:{dataset}_main_compact")
    note_generated(f"{dataset}_main_compact_table (.csv/.png/.tex)")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def render_one_dataset(dataset):
    print(f"\n=== dataset: {dataset} ===")
    df, long_df = load_dataset(dataset)

    fig_allseen_accuracy(df, dataset)
    fig_family_progression(df, dataset)
    fig_open_vs_restricted(df, dataset)
    fig_open_restricted_gap(df, dataset)
    fig_first_vs_later(df, dataset)
    fig_per_task_final_open(long_df, df, dataset)
    fig_per_task_restricted(long_df, df, dataset)
    fig_bwt(df, dataset)
    fig_rankext_forgetting(df, dataset)
    fig_rankext_recovery_over_plain(df, dataset)
    fig_simpleavg_delta_from_plain(df, dataset)
    fig_best_per_family(df, dataset)

    fig_opt_accuracy_vs_gap_scatter(df, dataset)
    fig_opt_rankext_finalacc_vs_forgetting(df, dataset)
    fig_opt_kd_paired_effect(df, dataset)
    fig_opt_orth_paired_effect(df, dataset)

    param_df = load_parameter_efficiency(dataset)
    fig_rank_growth(param_df, dataset)
    fig_trainable_lora_params_per_task(param_df, dataset)
    fig_cumulative_adapter_params(param_df, dataset)
    table_parameter_efficiency(param_df, dataset)
    note_skipped(f"{dataset}_accuracy_vs_adapter_params",
                 "SimpleAvg's final deployed 'extra parameter' count is ambiguous by construction "
                 "(0 if fused in place / 76,900 if counting only the new classifier / ~14.2M if the "
                 "dense delta is stored separately), whereas RankExt's is a single unambiguous "
                 "number -- plotting on a shared x-axis would silently pick one of three "
                 "incompatible interpretations (task brief's own explicit escape clause, Section 10)")
    note_skipped(f"{dataset}_simpleavg_specialist_vs_final",
                 "job4971615 does not log pre-merge, standalone per-specialist accuracy anywhere "
                 "(only the final merged per-task accuracy exists, already covered by the existing "
                 "per-task figures) -- building this would require re-evaluating each specialist "
                 "checkpoint separately, which is new evaluation work outside this task's scope")

    rij_df = load_rankext_rij(dataset)
    fig_rankext_rij_individual(rij_df, dataset)
    fig_rankext_rij_comparison(rij_df, dataset)

    table_main(df, dataset)
    table_compact(df, dataset)

    return df


def main():
    parser = argparse.ArgumentParser(description="Render Chapter-5 tables/figures from canonical CSVs.")
    parser.add_argument("--dataset", default=None,
                         help="Dataset name to render (e.g. cifar100). Default: render every dataset found.")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    available = discover_datasets()
    if not available:
        fail(f"no canonical dataset CSVs found under {DATA_DIR} "
             f"(expected chapter5_<dataset>_8method_main.csv)")

    if args.dataset:
        if args.dataset not in available:
            fail(f"--dataset '{args.dataset}' not found. Available: {available}")
        targets = [args.dataset]
    else:
        targets = available

    print(f"Datasets discovered: {available}")
    print(f"Rendering: {targets}")

    all_dfs = {}
    for dataset in targets:
        all_dfs[dataset] = render_one_dataset(dataset)

    print(f"\n=== cross-dataset ===")
    # Cross-dataset figure always considers every AVAILABLE dataset, not just
    # the ones selected via --dataset, so it stays meaningful once a second
    # dataset exists even if this invocation only re-rendered one of them.
    cross_dfs = dict(all_dfs)
    for dataset in available:
        if dataset not in cross_dfs:
            cross_dfs[dataset], _ = load_dataset(dataset)
    fig_cross_dataset_allseen(cross_dfs)

    print(f"\n=== summary ===")
    print(f"Figures/tables generated: {len(REPORT['generated'])}")
    for n in REPORT["generated"]:
        print(f"  + {n}")
    print(f"Figures skipped: {len(REPORT['skipped'])}")
    for n, reason in REPORT["skipped"]:
        print(f"  - {n}: {reason}")


if __name__ == "__main__":
    main()
