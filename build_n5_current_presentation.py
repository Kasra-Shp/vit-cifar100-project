from __future__ import annotations

import io
import math
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\ASUSCenter\Desktop\vit-cifar100-project")
TEMPLATE_PPTX = ROOT / "n5_methods_explanation.pptx"
RUN_PARENT = ROOT / "R1" / "full_comparison_20260622_144842"
RUN_DIRS = [p for p in RUN_PARENT.iterdir() if p.is_dir()]
if not RUN_DIRS:
    raise SystemExit(f"No run directory found under {RUN_PARENT}")
RUN_DIR = RUN_DIRS[0]
TABLES_DIR = RUN_DIR / "tables"
PLOTS_DIR = RUN_DIR / "plots"
BUILD_DIR = ROOT / "presentation_build_n5_current"
SLIDES_DIR = BUILD_DIR / "slides"
OUTPUT_PPTX = ROOT / "n5_current_results_supervisor_deck_20260622_formula_style_v2.pptx"
R1_FORMULA_IMAGES = {
    "trace": ROOT / "R1" / "1.png",
    "rank_zero_old": ROOT / "R1" / "2.png",
    "kd": ROOT / "R1" / "3.png",
    "factor": ROOT / "R1" / "4.png",
}

SLIDE_W = 13.333
SLIDE_H = 7.5
BG = "#f8f4ec"
PANEL = "#fffdf8"
TITLE = "#14345c"
TEXT = "#232323"
MUTED = "#5b5b5b"
GREEN = "#2f7d32"
RED = "#b23a48"
BLUE = "#4c78a8"
ORANGE = "#f58518"
PURPLE = "#7a4b94"

EMUS_X = 12192000
EMUS_Y = 6858000
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def shorten(name: str) -> str:
    mapping = {
        "full_finetune": "full_finetune",
        "simple_avg": "simple_avg",
        "simple_avg_replay": "simple_avg_replay",
        "simple_avg_kd": "simple_avg_kd",
        "simple_avg_factor_orth": "simple_avg_orth",
        "simple_avg_factor_orth_kd": "simple_avg_orth_kd",
        "do_merging_simple": "do_merging_simple",
        "rank_extension": "rank_extension",
        "rank_extension_replay": "rankext_replay",
        "rank_extension_kd_only": "rankext_kd",
        "rank_extension_orth_delta_trace_lam_50": "rankext_trace_orth",
        "rank_extension_orth_delta_trace_lam_50_kd": "rankext_trace_orth_kd",
        "rank_extension_orth_factor_lam_50": "rankext_factor_orth",
        "rank_extension_orth_factor_lam_50_kd": "rankext_factor_orth_kd",
        "joint_upper_bound": "joint_upper_bound",
    }
    return mapping.get(name, name)


def compact_label(name: str) -> str:
    mapping = {
        "full_finetune": "full_ft",
        "simple_avg": "savg",
        "simple_avg_replay": "savg_replay",
        "simple_avg_kd": "savg_kd",
        "simple_avg_factor_orth": "savg_forth",
        "simple_avg_factor_orth_kd": "savg_forth_kd",
        "do_merging_simple": "do_merge",
        "rank_extension": "rxt_base",
        "rank_extension_replay": "rxt_replay",
        "rank_extension_kd_only": "rxt_kd",
        "rank_extension_orth_delta_trace_lam_50": "rxt_trace",
        "rank_extension_orth_delta_trace_lam_50_kd": "rxt_trace_kd",
        "rank_extension_orth_factor_lam_50": "rxt_forth",
        "rank_extension_orth_factor_lam_50_kd": "rxt_forth_kd",
        "joint_upper_bound": "joint",
    }
    return mapping.get(name, name)


def infer_family(method: str) -> str:
    if method.startswith("simple_avg"):
        return "simple"
    if method.startswith("rank_extension"):
        return "rankext"
    if method == "do_merging_simple":
        return "merge"
    if method == "joint_upper_bound":
        return "joint"
    if method == "full_finetune":
        return "ft"
    return "other"


def family_color(method: str) -> str:
    fam = infer_family(method)
    return {
        "simple": BLUE,
        "rankext": GREEN,
        "merge": ORANGE,
        "joint": "#1b5e20",
        "ft": "#8c564b",
        "other": "#777777",
    }[fam]


def setup_figure(title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=150, facecolor=BG)
    fig.text(0.04, 0.935, title, fontsize=22, fontweight="bold", color=TITLE)
    if subtitle:
        fig.text(0.04, 0.895, subtitle, fontsize=10.5, color=MUTED)
    fig.add_axes([0.03, 0.03, 0.94, 0.001], facecolor="#d9d1c7").axis("off")
    return fig


def panel(fig, rect):
    ax = fig.add_axes(rect, facecolor=PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def add_bullets(ax, title: str, bullets: Iterable[str], fontsize: float = 11.0, width: int = 58):
    ax.text(0.02, 0.95, title, va="top", ha="left", fontsize=13, fontweight="bold", color=TITLE)
    y = 0.86
    for bullet in bullets:
        txt = wrap(bullet, width)
        ax.text(0.04, y, f"- {txt}", va="top", ha="left", fontsize=fontsize, color=TEXT, linespacing=1.35)
        y -= 0.09 + 0.027 * txt.count("\n")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def add_table(ax, df: pd.DataFrame, title: str, font_size: float = 10.0, col_widths=None):
    ax.axis("off")
    ax.text(0.0, 1.02, title, va="bottom", ha="left", fontsize=13, fontweight="bold", color=TITLE, transform=ax.transAxes)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d8d0c6")
        if r == 0:
            cell.set_facecolor("#ece5da")
            cell.set_text_props(weight="bold", color=TITLE)
        else:
            cell.set_facecolor(PANEL if r % 2 else "#f3eee6")


def render_slide_01():
    fig = setup_figure(
        "Current n5 Methods, Equations, and Results",
        "Source notebook: vit_lora_cifar100_full5step_n5.ipynb | Latest run: 2026-06-22 | Built from saved tables only",
    )
    ax = panel(fig, [0.05, 0.12, 0.90, 0.72])
    ax.text(0.02, 0.82, "CLIP-ViT + LoRA continual learning on Split CIFAR-100", fontsize=24, fontweight="bold", color=TITLE)
    ax.text(0.02, 0.70, "This deck is restricted to the current n5 setup and the newest R1 results.", fontsize=14, color=TEXT)
    bullets = [
        "Methods covered with equations and notebook cell references, not code.",
        "Main comparison emphasis: simple average vs rank extension.",
        "Detailed focus: rank_extension_kd_only and rank_extension_orth_factor_lam_50_kd.",
        "Supervisor questions answered explicitly: rank growth notation, zero-old switch, KD KL form, factor_orth notation, principal-angle next idea.",
    ]
    y = 0.58
    for bullet in bullets:
        ax.text(0.04, y, f"- {bullet}", fontsize=13.5, color=TEXT, va="top")
        y -= 0.11
    ax.text(0.02, 0.08, f"Run folder: {RUN_DIR.name}", fontsize=10.5, color=MUTED)
    return fig


def render_slide_02(enabled_methods: list[str], summary_df: pd.DataFrame):
    fig = setup_figure("Experiment Setup and Current Active Methods")
    left = panel(fig, [0.05, 0.12, 0.42, 0.76])
    right = panel(fig, [0.50, 0.12, 0.45, 0.76])
    add_bullets(
        left,
        "Current n5 setup",
        [
            "Backbone: openai/clip-vit-base-patch16, vision encoder only.",
            "Benchmark: CIFAR-100 split into 5 steps of 20 classes each.",
            "LoRA target modules: q_proj and v_proj.",
            "Simple-average LoRA: rank r = 80, alpha = 160.",
            "Rank-extension schedule: [16, 32, 48, 64, 80]. Only the new rank block is trainable each step.",
            "Replay setting when enabled: 20 stored examples per old class.",
            "KD setting: lambda_kd = 1.0, temperature T = 2.0.",
            "Orth setting: lambda_orth = 50.0 for both delta_trace and factor_orth variants.",
        ],
        fontsize=10.8,
        width=48,
    )
    methods_wrapped = [
        f"{idx + 1}. {method}"
        for idx, method in enumerate(enabled_methods)
    ]
    add_bullets(right, "Methods present in the current run", methods_wrapped, fontsize=10.2, width=42)
    top3 = summary_df.sort_values("all_seen", ascending=False).head(3)[["method", "all_seen"]].copy()
    top3["all_seen"] = top3["all_seen"].map(lambda v: f"{v:.2f}%")
    right.text(0.02, 0.06, "Top-3 by all_seen in this run", fontsize=12, fontweight="bold", color=TITLE)
    right.text(0.04, 0.02, " | ".join(f"{r.method}: {r.all_seen}" for r in top3.itertuples()), fontsize=10.5, color=TEXT)
    return fig


def render_method_cards_slide(title: str, rows: list[tuple[str, list[str], str, str]]):
    fig = setup_figure(title, "Equations are written from the current notebook behavior; cell references point to the implementing cells.")
    left = panel(fig, [0.05, 0.12, 0.43, 0.76])
    right = panel(fig, [0.52, 0.12, 0.43, 0.76])
    split = math.ceil(len(rows) / 2)
    columns = [rows[:split], rows[split:]]
    for ax, col_rows in zip([left, right], columns):
        card_gap = 0.05
        usable_h = 0.90
        card_h = (usable_h - card_gap * (len(col_rows) - 1)) / max(len(col_rows), 1)
        y_top = 0.95
        for method, equation_lines, explanation, cells in col_rows:
            card = FancyBboxPatch(
                (0.015, y_top - card_h),
                0.97,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#ddd4c9",
                facecolor="#fcf9f3",
                transform=ax.transAxes,
            )
            ax.add_patch(card)
            ax.text(0.04, y_top - 0.03, method, va="top", ha="left", fontsize=12.4, fontweight="bold", color=TITLE)
            ax.text(0.96, y_top - 0.03, cells, va="top", ha="right", fontsize=10.0, color=MUTED)
            y = y_top - 0.10
            for line in equation_lines:
                ax.text(0.06, y, line, va="top", ha="left", fontsize=16.0, color=TEXT)
                y -= 0.085 + 0.018 * line.count("\n")
            ax.text(
                0.06,
                y - 0.01,
                wrap(f"Meaning: {explanation}", 48),
                va="top",
                ha="left",
                fontsize=10.2,
                color=MUTED,
                linespacing=1.35,
            )
            y_top -= card_h + card_gap
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    return fig


def render_formula_reference_slide():
    fig = setup_figure("Reference Formula Style From R1 Images", "These are the exact typed styles you pointed to in R1. The updated equation slides follow this presentation style.")
    boxes = [
        ("Trace orthogonality", R1_FORMULA_IMAGES["trace"], [0.05, 0.52, 0.42, 0.30]),
        ("Rank growth and zero-old notation", R1_FORMULA_IMAGES["rank_zero_old"], [0.53, 0.52, 0.40, 0.30]),
        ("KD notation", R1_FORMULA_IMAGES["kd"], [0.05, 0.12, 0.25, 0.30]),
        ("Factor-orth notation", R1_FORMULA_IMAGES["factor"], [0.37, 0.10, 0.28, 0.34]),
    ]
    for title, img_path, rect in boxes:
        ax = panel(fig, rect)
        ax.text(0.02, 0.96, title, va="top", ha="left", fontsize=12.5, fontweight="bold", color=TITLE)
        img = mpimg.imread(img_path)
        ax_img = fig.add_axes([rect[0] + 0.01, rect[1] + 0.03, rect[2] - 0.02, rect[3] - 0.09])
        ax_img.imshow(img)
        ax_img.axis("off")
    note_ax = panel(fig, [0.69, 0.12, 0.24, 0.30])
    add_bullets(
        note_ax,
        "Use in the deck",
        [
            "delta_trace slides now use typed math instead of code-like placeholders.",
            "KD slides explicitly show the student log-softmax / teacher softmax form used in the notebook.",
            "factor_orth slides use factor overlap notation, not a trace symbol.",
            "rank-extension slides keep the rank-growth and zero-old distinction explicit.",
        ],
        fontsize=10.1,
        width=32,
    )
    return fig


def render_final_summary_slide():
    fig = setup_figure("Supervisor Answers and Final Talking Points")
    left = panel(fig, [0.05, 0.14, 0.42, 0.72])
    right = panel(fig, [0.51, 0.14, 0.44, 0.72])
    add_bullets(
        left,
        "Direct answers",
        [
            "Rank-extension in current n5 is rank growth, not constant-rank stacking: A_t = [A_old; A_new,t], B_t = [B_old, B_new,t].",
            "zero_old_merge is only a diagnostic switch. It disables old slices in forward after copying them, but it is not the main method used in the current run.",
            "KD uses student_log_probs = log softmax(z_S/T) and teacher_probs = softmax(z_T/T), then KL(log p_S || p_T) through the PyTorch kl_div API.",
            "Current n5 fixes lambda_KD = 1.0 and T = 2.0; there is no hyperparameter search in this notebook version.",
            "factor_orth notation should not show a trace. The code computes squared factor overlaps in A and B.",
        ],
        fontsize=10.6,
        width=50,
    )
    add_bullets(
        right,
        "Suggested presentation framing",
        [
            "Best non-joint result in this run: rank_extension_replay = 71.67 all_seen.",
            "Best no-replay rank-extension result: rank_extension_orth_factor_lam_50_kd = 68.43 all_seen.",
            "Compared with simple_avg_factor_orth_kd = 60.66, the rank-extension factor_orth+KD variant is +7.77 percentage points.",
            "Be honest that the plain no-replay rank_extension baseline is still weak at 20.47, so the base method path still needs debugging.",
            "Good next Tuesday topic: principal-angle regularization as a next orthogonality hypothesis.",
        ],
        fontsize=10.6,
        width=50,
    )
    return fig


def render_slide_05(rank_structure_df: pd.DataFrame, frozen_df: pd.DataFrame):
    fig = setup_figure("Supervisor Q&A: Rank Extension Notation and Zero-Old Switch")
    left = panel(fig, [0.05, 0.12, 0.47, 0.76])
    right = panel(fig, [0.55, 0.12, 0.40, 0.76])
    bullets = [
        "Correct current notation for step t is rank growth, not constant-rank stacking.",
        "A_t = [A_old; A_new^(t)] and B_t = [B_old, B_new^(t)], so the total rank increases from r_(t-1) to r_t.",
        "Current schedule in n5: 16 -> 32 -> 48 -> 64 -> 80.",
        "Only A_new^(t) and B_new^(t) are trainable; copied blocks A_old and B_old are frozen.",
        "In the main rank-extension path, old_active_in_forward = True, so copied old slices still contribute to the forward pass.",
        "zero_old_merge is only a diagnostic switch. When enabled for t > 1, the old copied slices are still copied and frozen, but they are disabled in forward. It is not the main method in this run.",
    ]
    add_bullets(left, "Answer to the notation question", bullets, fontsize=10.8, width=50)

    rs = rank_structure_df.iloc[0]
    max_a = frozen_df["A_max_abs_diff"].max() if not frozen_df.empty else np.nan
    max_b = frozen_df["B_max_abs_diff"].max() if not frozen_df.empty else np.nan
    evidence = pd.DataFrame(
        [
            ["Step 5 total rank", int(rs["total_rank"])],
            ["Step 5 frozen rank", int(rs["frozen_rank"])],
            ["Step 5 new rank", int(rs["new_rank"])],
            ["old_active_in_forward", bool(rs["old_active_in_forward"])],
            ["Target layers checked", int(len(rank_structure_df))],
            ["Max frozen A diff", f"{max_a:.10f}"],
            ["Max frozen B diff", f"{max_b:.10f}"],
        ],
        columns=["Diagnostic", "Value"],
    )
    ax_tbl = fig.add_axes([0.58, 0.48, 0.34, 0.28])
    add_table(ax_tbl, evidence, "Step-5 evidence from saved diagnostics", font_size=10.0, col_widths=[0.58, 0.32])
    right.text(0.02, 0.36, "Interpretation", fontsize=13, fontweight="bold", color=TITLE)
    right.text(
        0.04,
        0.31,
        wrap(
            "The current implementation follows growing-rank construction in Cell 16. The code copies previous A/B blocks into a larger LoRA, freezes them, and keeps only the newly added slice trainable. The zero_old_merge variants are separate diagnostics, not the definition of rank extension.",
            48,
        ),
        fontsize=10.8,
        color=TEXT,
        va="top",
        linespacing=1.35,
    )
    return fig


def render_slide_06():
    fig = setup_figure("Supervisor Q&A: KD, Orthogonality, and Principal Angles")
    left = panel(fig, [0.05, 0.12, 0.44, 0.76])
    right = panel(fig, [0.52, 0.12, 0.43, 0.76])
    add_bullets(
        left,
        "KD answer",
        [
            "Current code uses student_log_probs = log softmax(z_s / T) and teacher_probs = softmax(z_t / T), then F.kl_div(student_log_probs, teacher_probs) * T^2.",
            "Because of the PyTorch API, this computes KL(P_teacher^(T) || P_student^(T)). The log on the student side is API format, not a different mathematical target.",
            "Current n5 does not do a hyperparameter search for KD. The notebook fixes T = 2 and lambda_kd = 1 for comparability across KD variants.",
            "Cells: simple-average KD path in Cell 11; rank-extension KD path in Cell 16; hyperparameters in Cell 3.",
        ],
        fontsize=10.8,
        width=48,
    )
    add_bullets(
        right,
        "Orth answer",
        [
            "delta_trace in current rank-extension uses mean_l | < Delta W_old^l, Delta W_new^l > |, with diagnostics also logging the normalized squared version.",
            "factor_orth in current code does not use a trace. It uses || A_old_hat A_new_hat^T ||_F^2 + || B_old_hat^T B_new_hat ||_F^2 over each layer.",
            "So if factor_orth notation still shows a trace, the notation is misleading and should be corrected.",
            "Principal-angle next idea for Tuesday: build orthonormal bases Q_old and Q_new, compute G = Q_old^T Q_new, and penalize the singular values sigma_k = cos(theta_k) or their squares. This matches the principal-angle definition in sparse-plex.",
            "Source for the principal-angle definition: https://sparse-plex.readthedocs.io/en/latest/book/linear_algebra/principal_angles.html",
        ],
        fontsize=10.3,
        width=49,
    )
    return fig


def render_slide_07():
    fig = setup_figure("Overall Results: All Methods in the Current Run")
    ax1 = fig.add_axes([0.05, 0.14, 0.44, 0.68])
    ax2 = fig.add_axes([0.53, 0.14, 0.42, 0.68])
    img1 = mpimg.imread(PLOTS_DIR / "01_grouped_accuracy_comparison_all_methods.png")
    img2 = mpimg.imread(PLOTS_DIR / "02_ranking_by_all_seen_all_methods.png")
    ax1.imshow(img1)
    ax1.axis("off")
    ax2.imshow(img2)
    ax2.axis("off")
    fig.text(0.05, 0.085, "Left: final first_step / later_steps / all_seen for all methods. Right: ranking by all_seen.", fontsize=10.5, color=MUTED)
    return fig


def render_slide_08(summary_df: pd.DataFrame):
    fig = setup_figure("Main Comparison: Simple Average vs Rank Extension")
    ax = fig.add_axes([0.06, 0.16, 0.58, 0.66], facecolor=PANEL)
    focus_pairs = [
        ("No replay", "simple_avg", "rank_extension"),
        ("Replay", "simple_avg_replay", "rank_extension_replay"),
        ("KD only", "simple_avg_kd", "rank_extension_kd_only"),
        ("Factor orth", "simple_avg_factor_orth", "rank_extension_orth_factor_lam_50"),
        ("Factor orth + KD", "simple_avg_factor_orth_kd", "rank_extension_orth_factor_lam_50_kd"),
    ]
    labels = []
    simple_vals = []
    rank_vals = []
    for label, s_method, r_method in focus_pairs:
        labels.append(label)
        simple_vals.append(float(summary_df.loc[s_method, "all_seen"]))
        rank_vals.append(float(summary_df.loc[r_method, "all_seen"]))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, simple_vals, width, color=BLUE, label="simple_avg family")
    ax.bar(x + width / 2, rank_vals, width, color=GREEN, label="rank_extension family")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("All-seen accuracy (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    for idx, (sv, rv) in enumerate(zip(simple_vals, rank_vals)):
        ax.text(idx - width / 2, sv + 1.0, f"{sv:.2f}", ha="center", fontsize=9)
        ax.text(idx + width / 2, rv + 1.0, f"{rv:.2f}", ha="center", fontsize=9)

    df = pd.DataFrame(
        {
            "Setting": labels,
            "simple_avg all_seen": [f"{v:.2f}%" for v in simple_vals],
            "rank_extension all_seen": [f"{v:.2f}%" for v in rank_vals],
            "Rankext - Simple": [f"{(rv - sv):+.2f} pp" for sv, rv in zip(simple_vals, rank_vals)],
        }
    )
    ax_tbl = fig.add_axes([0.68, 0.18, 0.27, 0.60])
    add_table(ax_tbl, df, "Exact current-run numbers", font_size=9.8, col_widths=[0.30, 0.24, 0.24, 0.19])
    fig.text(
        0.06,
        0.08,
        "Main pattern: no-replay favors simple_avg, while replay, KD-only, and factor_orth+KD favor rank_extension.",
        fontsize=10.5,
        color=MUTED,
    )
    return fig


def render_slide_09(summary_df: pd.DataFrame):
    fig = setup_figure("Retention / Plasticity Trade-off in the Focused Methods")
    ax_scatter = fig.add_axes([0.05, 0.16, 0.45, 0.66], facecolor=PANEL)
    img = mpimg.imread(PLOTS_DIR / "03_retention_vs_plasticity_scatter.png")
    ax_scatter.imshow(img)
    ax_scatter.axis("off")

    focus_methods = [
        "simple_avg",
        "simple_avg_replay",
        "simple_avg_kd",
        "simple_avg_factor_orth",
        "simple_avg_factor_orth_kd",
        "rank_extension",
        "rank_extension_replay",
        "rank_extension_kd_only",
        "rank_extension_orth_factor_lam_50",
        "rank_extension_orth_factor_lam_50_kd",
    ]
    table_df = summary_df.loc[focus_methods, ["first_step", "later_steps", "all_seen", "old_new_gap"]].copy()
    table_df.insert(0, "method", [compact_label(m) for m in table_df.index])
    for col in ["first_step", "later_steps", "all_seen", "old_new_gap"]:
        table_df[col] = table_df[col].map(lambda v: f"{float(v):.2f}")
    table_df = table_df.reset_index(drop=True)
    ax_tbl = fig.add_axes([0.54, 0.14, 0.41, 0.68])
    add_table(ax_tbl, table_df, "Focused metric table", font_size=9.0, col_widths=[0.28, 0.14, 0.14, 0.14, 0.14])
    fig.text(
        0.54,
        0.105,
        wrap("Warning sign: savg_forth and savg_forth_kd get high later-step scores but collapse first-step retention.", 56),
        fontsize=10.0,
        color=RED,
    )
    return fig


def render_slide_10(summary_df: pd.DataFrame, diag_df: pd.DataFrame):
    fig = setup_figure("Why Factor-Orth Behaves So Differently in This Run")
    methods = [
        "simple_avg_factor_orth",
        "simple_avg_factor_orth_kd",
        "rank_extension_orth_factor_lam_50",
        "rank_extension_orth_factor_lam_50_kd",
        "rank_extension_orth_delta_trace_lam_50",
        "rank_extension_orth_delta_trace_lam_50_kd",
    ]
    plot_df = summary_df.loc[methods, ["all_seen"]].copy()
    weighted = {}
    kd_ratio = {}
    for method in methods:
        rows = diag_df[diag_df["method"] == method]
        weighted[method] = float(rows["weighted_orth_over_CE"].max()) if len(rows) else np.nan
        kd_ratio[method] = float(rows["kd_over_CE"].max()) if len(rows) else np.nan
    plot_df["weighted_orth_over_CE"] = pd.Series(weighted)
    plot_df["kd_over_CE"] = pd.Series(kd_ratio)
    labels = [compact_label(m) for m in methods]

    ax1 = fig.add_axes([0.12, 0.18, 0.34, 0.62], facecolor=PANEL)
    ax2 = fig.add_axes([0.54, 0.18, 0.18, 0.62], facecolor=PANEL)
    ax3 = fig.add_axes([0.78, 0.18, 0.16, 0.62], facecolor=PANEL)

    ax1.barh(labels, plot_df["all_seen"], color=[family_color(m) for m in methods])
    for y, val in enumerate(plot_df["all_seen"]):
        ax1.text(float(val) + 0.8, y, f"{float(val):.2f}", va="center", fontsize=9)
    ax1.set_xlabel("All-seen (%)")
    ax1.set_title("Accuracy outcome")
    ax1.grid(axis="x", alpha=0.22)
    ax1.tick_params(axis="y", labelsize=9.2)

    ax2.barh(labels, plot_df["weighted_orth_over_CE"], color=ORANGE)
    for y, val in enumerate(plot_df["weighted_orth_over_CE"]):
        ax2.text(float(val) + 0.6, y, f"{float(val):.2f}", va="center", fontsize=8.5)
    ax2.set_xlabel("weighted orth / CE")
    ax2.set_title("Regularizer scale")
    ax2.grid(axis="x", alpha=0.22)
    ax2.tick_params(axis="y", labelsize=9.2)

    ax3.barh(labels, plot_df["kd_over_CE"], color=PURPLE)
    for y, val in enumerate(plot_df["kd_over_CE"]):
        if math.isfinite(float(val)):
            ax3.text(float(val) + 0.05, y, f"{float(val):.2f}", va="center", fontsize=8.5)
    ax3.set_xlabel("KD / CE")
    ax3.set_title("KD scale")
    ax3.grid(axis="x", alpha=0.22)
    ax3.tick_params(axis="y", labelsize=9.2)

    fig.text(
        0.06,
        0.08,
        wrap("Key observation: simple_avg factor_orth runs with weighted orth / CE above 100x, while rank_extension factor_orth+KD stays below 1x. This likely explains why factor_orth is destructive for simple_avg but still usable in rank_extension.", 130),
        fontsize=9.9,
        color=MUTED,
    )
    return fig


def render_slide_11(summary_df: pd.DataFrame):
    fig = setup_figure("Focused Outcome: RankExt + KD and RankExt + Factor-Orth + KD")
    left = panel(fig, [0.05, 0.14, 0.43, 0.72])
    right = panel(fig, [0.52, 0.14, 0.43, 0.72])
    methods = [
        "simple_avg_replay",
        "rank_extension_replay",
        "rank_extension_kd_only",
        "rank_extension_orth_factor_lam_50_kd",
        "joint_upper_bound",
    ]
    df = summary_df.loc[methods, ["first_step", "later_steps", "all_seen", "gap_to_joint"]].copy()
    df.insert(0, "method", [shorten(m) for m in df.index])
    for col in ["first_step", "later_steps", "all_seen", "gap_to_joint"]:
        df[col] = df[col].map(lambda v: f"{float(v):.2f}")
    add_table(left, df.reset_index(drop=True), "Methods to emphasize in the oral presentation", font_size=10.0, col_widths=[0.30, 0.14, 0.14, 0.14, 0.16])

    bullets = [
        "rank_extension_replay is the strongest non-joint method in this run: all_seen = 71.67%.",
        "rank_extension_orth_factor_lam_50_kd is the best no-replay rank-extension variant: all_seen = 68.43%, first_step = 80.00%.",
        "Compared with simple_avg_factor_orth_kd (60.66%), rank_extension_orth_factor_lam_50_kd is +7.77 percentage points on all_seen.",
        "Compared with simple_avg_kd (39.16%), rank_extension_kd_only is +13.71 percentage points on all_seen.",
        "The plain no-replay rank_extension baseline is still weak at 20.47%, so the final narrative should acknowledge that the base rank-extension path still needs debugging.",
    ]
    add_bullets(right, "How to frame these results", bullets, fontsize=10.9, width=47)
    return fig


def render_slide_12():
    fig = setup_figure("Conclusions and Tuesday Discussion Points")
    left = panel(fig, [0.05, 0.14, 0.42, 0.72])
    right = panel(fig, [0.51, 0.14, 0.44, 0.72])
    add_bullets(
        left,
        "What is defensible now",
        [
            "The current code path in Cell 16 is a true growing-rank construction, not constant-rank stacking.",
            "The frozen copied blocks are actually frozen in the saved diagnostics: max A/B change = 0 at step 5.",
            "The KD term is implemented as KL(P_teacher^(T) || P_student^(T)) via PyTorch kl_div with student log-probs and teacher probs.",
            "Among the currently saved no-replay rank-extension methods, rank_extension_orth_factor_lam_50_kd is the strongest.",
        ],
        fontsize=10.8,
        width=48,
    )
    add_bullets(
        right,
        "What still needs discussion",
        [
            "The plain rank_extension no-replay baseline is too weak to call the current comparison fully settled.",
            "factor_orth notation should be updated in the slides to remove any trace symbol; the code uses factor overlap penalties instead.",
            "The KD hyperparameters are fixed, not tuned. If the supervisor asks, say this explicitly.",
            "Principal-angle regularization is a reasonable next hypothesis: derive orthonormal bases Q_old and Q_new, compute G = Q_old^T Q_new, then penalize sigma_max(G) or sum_k sigma_k^2 = sum_k cos^2(theta_k).",
            "Good next discussion question: should principal angles be applied to delta subspaces, factor subspaces, or both?",
        ],
        fontsize=10.5,
        width=49,
    )
    return fig


def save_slide(fig, idx: int):
    path = SLIDES_DIR / f"slide_{idx:02d}.jpg"
    fig.savefig(path, dpi=150, bbox_inches=None, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def make_slide_xml(image_rel_id: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="SlideImage"/>
          <p:cNvPicPr>
            <a:picLocks noChangeAspect="1"/>
          </p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="{image_rel_id}"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{EMUS_X}" cy="{EMUS_Y}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
""".encode("utf-8")


def updated_rels_xml(original: bytes, media_target: str) -> bytes:
    ET.register_namespace("", REL_NS)
    root = ET.fromstring(original)
    ids = []
    for child in root.findall(f"{{{REL_NS}}}Relationship"):
        rid = child.attrib["Id"]
        if rid.startswith("rId"):
            try:
                ids.append(int(rid[3:]))
            except ValueError:
                pass
    new_id = f"rId{(max(ids) + 1) if ids else 1}"
    ET.SubElement(
        root,
        f"{{{REL_NS}}}Relationship",
        {
            "Id": new_id,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": media_target,
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), new_id


def build_pptx(slide_paths: list[Path]):
    with zipfile.ZipFile(TEMPLATE_PPTX, "r") as zin, zipfile.ZipFile(OUTPUT_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        skip = set()
        for i in range(1, len(slide_paths) + 1):
            skip.add(f"ppt/slides/slide{i}.xml")
            skip.add(f"ppt/slides/_rels/slide{i}.xml.rels")
        for item in zin.infolist():
            if item.filename in skip:
                continue
            zout.writestr(item, zin.read(item.filename))

        for idx, slide_path in enumerate(slide_paths, start=1):
            original_rels = zin.read(f"ppt/slides/_rels/slide{idx}.xml.rels")
            rels_xml, image_rel_id = updated_rels_xml(original_rels, f"../media/generated_slide_{idx:02d}.jpeg")
            zout.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", rels_xml)
            zout.writestr(f"ppt/slides/slide{idx}.xml", make_slide_xml(image_rel_id))
            zout.writestr(f"ppt/media/generated_slide_{idx:02d}.jpeg", slide_path.read_bytes())


def main():
    BUILD_DIR.mkdir(exist_ok=True)
    SLIDES_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "mathtext.fontset": "stix",
    })

    summary_df = pd.read_csv(TABLES_DIR / "summary_metrics_clip_vit_percent.csv", index_col=0)
    comparison_df = pd.read_csv(TABLES_DIR / "comparison_overview_clip_vit_percent.csv")
    diag_df = pd.read_csv(TABLES_DIR / "orth_kd_diagnostics_summary_by_method.csv")
    rank_structure_df = pd.read_csv(TABLES_DIR / "rank_extension_step_5_rank_structure.csv")
    frozen_df = pd.read_csv(TABLES_DIR / "rank_extension_step_5_frozen_rank_blocks.csv")

    enabled_methods = summary_df.index.tolist()

    method_rows_a = [
        ("full_finetune", [r"$\theta_t^\star=\arg\min_{\theta}\;\mathbb{E}_{(x,y)\in D_t}\,\mathrm{CE}(f_{\theta}(x),y)$"], "Full backbone adaptation on the current step only; this is the sequential fine-tuning baseline.", "C10"),
        ("simple_avg", [r"$\Delta W_t=s_t B_tA_t$", r"$\bar{\Delta W}=\frac{1}{T}\sum_{t=1}^{T}\Delta W_t$", r"$W^\star=W_0+\bar{\Delta W}$"], "Train one LoRA per step, then average the final step deltas and merge once into the backbone.", "C9, C12"),
        ("simple_avg_replay", [r"$D_t^\prime=D_t\cup R_{<t}$", r"$\bar{\Delta W}_{replay}=\frac{1}{T}\sum_{t=1}^{T}\Delta W_t^\prime$"], "Same simple average, but each later step is trained with replayed old samples before averaging.", "C7, C13"),
        ("do_merging_simple", [r"$\Delta W_t^{(\ell)}\rightarrow\left(M_t^{(\ell)},D_t^{(\ell)}\right)$", "orthogonalize task directions by Gram-Schmidt", r"$\Delta W_{merge}^{(\ell)}=\mathrm{merge}\!\left(M_t^{(\ell)},\widetilde{D}_t^{(\ell)}\right)$"], "Decompose each layer update into magnitude and direction, orthogonalize directions, then merge the corrected updates.", "C9, C14"),
    ]
    method_rows_b = [
        ("simple_avg_kd", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{KD}T^2\,\mathrm{KL}\!\left(P_T^{(T)}\,\|\,P_S^{(T)}\right)$", r"$W^\star=W_0+\bar{\Delta W}_{KD}$"], "Knowledge distillation is added during each step, then the resulting LoRA deltas are averaged as usual.", "C11, C12"),
        ("simple_avg_factor_orth", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\cdot\frac{1}{L}\sum_{\ell=1}^{L}\mathcal{L}_{factor}^{(\ell)}$", r"$\mathcal{L}_{factor}^{(\ell)}=\|\hat A_{ref}^{(\ell)}(\hat A_t^{(\ell)})^T\|_F^2+\|(\hat B_{ref}^{(\ell)})^T\hat B_t^{(\ell)}\|_F^2$"], "Penalize overlap between old and current LoRA factor spaces while keeping the standard LoRA forward update.", "C11, C12"),
        ("simple_avg_factor_orth_kd", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\mathcal{L}_{factor}+\lambda_{KD}T^2\,\mathrm{KL}\!\left(P_T^{(T)}\,\|\,P_S^{(T)}\right)$"], "Combine factor-space orthogonality and KD before averaging the independent LoRA updates.", "C11, C12"),
        ("rank_extension", [r"$A_t=[A_{old};A_{new,t}],\qquad B_t=[B_{old},B_{new,t}]$", r"$\Delta W_t=s_t B_tA_t$", r"$\mathrm{trainable:}\;A_{new,t},B_{new,t}\;\mathrm{only}$"], "One growing LoRA: copy old slices, freeze them, and train only the newly added rank block at each step.", "C16"),
    ]
    method_rows_c = [
        ("rank_extension_replay", [r"$D_t^\prime=D_t\cup R_{<t}$", r"$A_t=[A_{old};A_{new,t}],\;B_t=[B_{old},B_{new,t}]$"], "Growing-rank LoRA with replay added to later steps, while preserving and freezing copied old slices.", "C7, C16"),
        ("rank_extension_kd_only", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{KD}T^2\,\mathrm{KL}\!\left(P_T^{(T)}\,\|\,P_S^{(T)}\right)$"], "Keep the growing-rank construction and add only the KD term during training of the new rank block.", "C16"),
        ("rank_extension_orth_delta_trace_lam_50", [r"$\mathcal{L}_{trace}=\frac{1}{L}\sum_{\ell=1}^{L}\left|\mathrm{tr}\!\left((\Delta W_{old}^{(\ell)})^{T}\Delta W_{new}^{(\ell)}\right)\right|$", r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\mathcal{L}_{trace}$"], "Penalize direct overlap between the old frozen delta and the newly learned delta in each layer.", "C15, C16"),
        ("rank_extension_orth_delta_trace_lam_50_kd", [r"$\mathcal{L}_{trace}=\frac{1}{L}\sum_{\ell=1}^{L}\left|\mathrm{tr}\!\left((\Delta W_{old}^{(\ell)})^{T}\Delta W_{new}^{(\ell)}\right)\right|$", r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\mathcal{L}_{trace}+\lambda_{KD}T^2\,\mathrm{KL}\!\left(P_T^{(T)}\,\|\,P_S^{(T)}\right)$"], "Use the same exact trace-overlap penalty, then add KD on top of it.", "C15, C16"),
    ]
    method_rows_d = [
        ("rank_extension_orth_factor_lam_50", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\cdot\frac{1}{L}\sum_{\ell=1}^{L}\mathcal{L}_{factor}^{(\ell)}$", r"$\mathcal{L}_{factor}^{(\ell)}=\|\hat A_{old}^{(\ell)}(\hat A_{new}^{(\ell)})^T\|_F^2+\|(\hat B_{old}^{(\ell)})^T\hat B_{new}^{(\ell)}\|_F^2$"], "Constrain overlap in the factor spaces of the frozen and new rank blocks instead of using a trace penalty.", "C16"),
        ("rank_extension_orth_factor_lam_50_kd", [r"$\mathcal{L}=\mathcal{L}_{CE}+\lambda_{orth}\mathcal{L}_{factor}+\lambda_{KD}T^2\,\mathrm{KL}\!\left(P_T^{(T)}\,\|\,P_S^{(T)}\right)$"], "The strongest saved no-replay rank-extension variant: factor-space orthogonality plus KD.", "C16"),
        ("joint_upper_bound", [r"$\theta_{joint}^\star=\arg\min_{\theta}\;\mathbb{E}_{(x,y)\in\cup_t D_t}\,\mathrm{CE}(f_{\theta}(x),y)$"], "Upper bound from joint training on all data together; used only as a reference target.", "C17"),
    ]

    slide_paths = [
        save_slide(render_slide_01(), 1),
        save_slide(render_slide_02(enabled_methods, summary_df.reset_index().rename(columns={"index": "method"})), 2),
        save_slide(render_method_cards_slide("Method Equations: Baselines, Simple Average, and DO-Merging", method_rows_a), 3),
        save_slide(render_method_cards_slide("Method Equations: KD, Factor-Orth, and Rank-Extension Base", method_rows_b), 4),
        save_slide(render_method_cards_slide("Method Equations: Rank-Extension Replay, KD, and Delta-Trace", method_rows_c), 5),
        save_slide(render_method_cards_slide("Method Equations: Rank-Extension Factor-Orth and Joint Upper Bound", method_rows_d), 6),
        save_slide(render_formula_reference_slide(), 7),
        save_slide(render_slide_07(), 8),
        save_slide(render_slide_08(summary_df), 9),
        save_slide(render_slide_09(summary_df), 10),
        save_slide(render_slide_10(summary_df, diag_df), 11),
        save_slide(render_final_summary_slide(), 12),
    ]

    build_pptx(slide_paths)
    print(f"Built presentation: {OUTPUT_PPTX}")
    print(f"Slide images: {SLIDES_DIR}")


if __name__ == "__main__":
    main()
