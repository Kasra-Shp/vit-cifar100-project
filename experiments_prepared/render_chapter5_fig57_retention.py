"""Render thesis Figure 5.7: RankExt R_ij trajectories plus SimpleAvg final
per-step accuracies (CIFAR-100 5x20, seed 42, job 4971615).

Inputs (read-only, produced by build_chapter5_*_data.py):
  R7/chapter5_main_benchmark/data/chapter5_cifar100_rankext_Rij.csv
      in-training evaluations (before classifier rescaling), RankExt only
  R7/chapter5_main_benchmark/data/chapter5_cifar100_per_task_metrics.csv
      final-model per-step open accuracy (after classifier rescaling), all 8 methods

SimpleAvg has no persistent intermediate merged model, so only its final
per-step row is drawn; no intermediate SimpleAvg values are created.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "R7", "chapter5_main_benchmark", "data")
OUT = os.path.join(ROOT, "thesis_writing", "thesis_latex", "figures", "chapter5",
                   "cifar100_retention_rankext_Rij_simpleavg_final")

RANKEXT = [("rankext", "RankExt"),
           ("rankext_kd_protect", "RankExt + KD + Protect"),
           ("rankext_factororth", "RankExt + FactorOrth"),
           ("rankext_factororth_kd_protect", "RankExt + FactorOrth\n+ KD + Protect")]
SIMPLEAVG = [("simple_avg", "SimpleAvg"),
             ("simple_avg_kd", "SimpleAvg + KD"),
             ("simple_avg_denseorth", "SimpleAvg + FactorOrth"),
             ("simple_avg_denseorth_kd", "SimpleAvg + FactorOrth\n+ KD")]

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
rij = pd.read_csv(os.path.join(DATA, "chapter5_cifar100_rankext_Rij.csv"))
final = pd.read_csv(os.path.join(DATA, "chapter5_cifar100_per_task_metrics.csv"))
final = final[final.metric == "open_accuracy"]


def final_row(mid):
    sub = final[final.method_id == mid].sort_values("task")
    assert list(sub.task) == [1, 2, 3, 4, 5], mid
    return sub.value.to_numpy(float)


def rankext_matrix(mid):
    sub = rij[(rij.method_id == mid) & (rij.metric == "open_accuracy")]
    mat = np.full((7, 5), np.nan)          # rows 0-4: R_ij, row 5: gap, row 6: final
    for r in sub.itertuples():
        mat[r.train_step - 1, r.eval_task - 1] = r.accuracy
    mat[6] = final_row(mid)
    return mat


cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("white")


def draw(ax, mat, ylabels):
    im = ax.imshow(np.ma.masked_invalid(mat), vmin=0, vmax=100, cmap=cmap, aspect="equal")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v < 50 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"S{j}" for j in range(1, 6)], fontsize=8.5)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


fig = plt.figure(figsize=(10.0, 4.9))
gs = fig.add_gridspec(2, 4, height_ratios=[7, 1.0], hspace=0.42, wspace=0.18,
                      left=0.075, right=0.90, top=0.93, bottom=0.07)
im = None
for k, (mid, title) in enumerate(RANKEXT):
    ax = fig.add_subplot(gs[0, k])
    im = draw(ax, rankext_matrix(mid),
              [f"i={i}" for i in range(1, 6)] + ["", "Final"] if k == 0 else [""] * 7)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("Evaluated step group j", fontsize=8.5)
fig.axes[0].set_ylabel("RankExt: trained through step i", fontsize=9)
for k, (mid, title) in enumerate(SIMPLEAVG):
    ax = fig.add_subplot(gs[1, k])
    draw(ax, final_row(mid)[None, :], ["Final"] if k == 0 else [""])
    ax.set_title(title, fontsize=9.5)
    if k == 0:
        ax.set_ylabel("SimpleAvg", fontsize=9)
cax = fig.add_axes([0.925, 0.12, 0.015, 0.76])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Open (all-seen) accuracy (%)", fontsize=9)
cb.ax.tick_params(labelsize=8.5)
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("wrote", OUT + ".pdf")
