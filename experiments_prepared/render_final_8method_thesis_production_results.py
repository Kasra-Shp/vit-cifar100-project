"""
Final 8-method results table/PNG renderer for the FINAL 8-METHOD THESIS
PRODUCTION experiment (experiments_prepared/final_8method_thesis_production.py).

Standalone, isolated module -- does not import from or modify
final_8method_thesis_production.py, final_9method_5x20_performance_recovery.py,
final_8method_5x20_refined.py, or render_final_8method_results.py. A new
module (not a reuse of render_final_8method_results.py) is required because
that module's DISPLAY_ROWS hardcodes the OLD lam20 SimpleAvg-DenseOrth method
identifiers, which no longer exist in this production run's active method
set (renamed lam20 -> lam1, Section 7 of the task brief).

The production script imports and calls
`render_final_8method_thesis_production_results()` from its own reporting
section, once, only after verifying all 8 expected methods are present (see
validate_summary_before_render() below) -- if validation fails, this module
raises loudly instead of silently producing a partial/misleading table.

Reads values programmatically from the actual summary dataframe/CSV -- never
hard-codes a result number.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Exact 8 rows, in the required display order (Section 20 of the task
# brief), mapping this codebase's internal method name -> the thesis-ready
# display label.
DISPLAY_ROWS = [
    ("simple_avg", "SimpleAvg"),
    ("simple_avg_kd_oldseen_T2_warmup", "SimpleAvg + KD"),
    ("simple_avg_dense_orth_lam1", "SimpleAvg + DenseOrth (λ=1)"),
    ("simple_avg_dense_orth_lam1_kd_oldseen_T2_warmup", "SimpleAvg + DenseOrth (λ=1) + KD"),
    ("rank_extension", "RankExt"),
    ("rank_extension_fullkd_T2_protect30", "RankExt + KD + Protect"),
    ("rank_extension_factor_orth_lam50", "RankExt + FactorOrth"),
    ("rank_extension_factor_orth_lam50_fullkd_T2_protect30", "RankExt + FactorOrth + KD + Protect"),
]
EXPECTED_INTERNAL_METHODS = [m for m, _ in DISPLAY_ROWS]


def validate_summary_before_render(summary_df, method_col="method"):
    """Fails loudly (raises) rather than silently rendering a misleading
    table if the summary is not exactly the expected 8 arms, each exactly
    once, T4 absent, no lam20-named arm, no unexpected extras (e.g. a stray
    calibration row)."""
    if method_col not in summary_df.columns:
        raise ValueError(f"summary_df is missing the '{method_col}' column: {list(summary_df.columns)}")

    present = list(summary_df[method_col])
    present_set = set(present)
    expected_set = set(EXPECTED_INTERNAL_METHODS)

    if len(present) != len(present_set):
        dupes = [m for m in present_set if present.count(m) > 1]
        raise RuntimeError(f"FAIL: duplicate method rows in summary: {dupes}")

    missing = expected_set - present_set
    if missing:
        raise RuntimeError(
            f"FAIL: {len(missing)} expected method(s) missing from summary -- refusing to render an "
            f"incomplete/misleading final table. Missing: {sorted(missing)}"
        )

    extra = present_set - expected_set
    if extra:
        raise RuntimeError(
            f"FAIL: unexpected extra row(s) in summary (e.g. a stray calibration-method row) -- "
            f"refusing to render. Extra: {sorted(extra)}. This table must contain exactly the 8 "
            f"canonical arms, never a separate calibration method/row."
        )

    if "simple_avg_kd_oldseen_T4_warmup" in present_set:
        raise RuntimeError("FAIL: T4 arm present in summary -- this experiment must not include it.")

    if any("lam20" in str(m) for m in present_set):
        raise RuntimeError(
            "FAIL: a lam20-named SimpleAvg-DenseOrth arm is present -- this production run must use "
            "lambda=1 (lam1) arms only."
        )

    if len(present) != 8:
        raise RuntimeError(f"FAIL: expected exactly 8 rows, got {len(present)}.")

    return True


def _fmt_pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}"


def _fmt_signed(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.3f}"


def build_display_table(summary_df, method_col="method", all_seen_col="all_seen",
                         restricted_col="restricted_mean", bwt_col="BWT", forgetting_col="forgetting"):
    """Builds the exact 8-row display dataframe (Method, All-seen Accuracy (%),
    Restricted Accuracy (%), BWT, Forgetting) from the real summary dataframe,
    in the required fixed order. No FWT column. No calibration gain/pre-
    calibration/learned-scale columns -- RankExt's All-seen is simply
    whatever value the summary dataframe already has for that method (the
    normal final result after its full, integrated evaluation/calibration
    pipeline -- historical row-norm calibration only, no task-scale
    calibration)."""
    validate_summary_before_render(summary_df, method_col=method_col)
    indexed = summary_df.set_index(method_col)

    rows = []
    for internal_name, display_name in DISPLAY_ROWS:
        row = indexed.loc[internal_name]
        all_seen = row.get(all_seen_col, np.nan)
        restricted = row.get(restricted_col, np.nan)
        bwt = row.get(bwt_col, np.nan)
        forgetting = row.get(forgetting_col, np.nan)
        rows.append({
            "Method": display_name,
            "All-seen Accuracy (%)": _fmt_pct(all_seen),
            "Restricted Accuracy (%)": _fmt_pct(restricted),
            "BWT": _fmt_signed(bwt),
            "Forgetting": _fmt_signed(forgetting),
            "_all_seen_numeric": float(all_seen) if not (all_seen is None or (isinstance(all_seen, float) and np.isnan(all_seen))) else float("-inf"),
        })
    return pd.DataFrame(rows)


def render_final_8method_thesis_production_results(
    summary_df, output_dir, method_col="method", all_seen_col="all_seen",
    restricted_col="restricted_mean", bwt_col="BWT", forgetting_col="forgetting",
    also_render_bar_chart=True,
):
    """Main entry point. Validates, builds the display table, writes the CSV
    (without the internal `_all_seen_numeric` sort/bold-helper column), and
    renders the mandatory table PNG (>=300 DPI) plus an optional all-seen bar
    chart. Returns (csv_path, table_png_path, bar_png_path_or_None)."""
    os.makedirs(output_dir, exist_ok=True)
    display_df = build_display_table(
        summary_df, method_col=method_col, all_seen_col=all_seen_col,
        restricted_col=restricted_col, bwt_col=bwt_col, forgetting_col=forgetting_col,
    )
    best_idx = int(display_df["_all_seen_numeric"].idxmax())
    csv_df = display_df.drop(columns=["_all_seen_numeric"])

    csv_path = os.path.join(output_dir, "final_8method_thesis_results_table.csv")
    csv_df.to_csv(csv_path, index=False, encoding="utf-8")

    table_png_path = os.path.join(output_dir, "final_8method_thesis_results_table.png")
    _render_table_png(csv_df, best_idx, table_png_path)

    bar_png_path = None
    if also_render_bar_chart:
        bar_png_path = os.path.join(output_dir, "final_8method_thesis_allseen_accuracy.png")
        _render_allseen_bar_png(display_df, best_idx, bar_png_path)

    return csv_path, table_png_path, bar_png_path


def _render_table_png(csv_df, best_idx, out_path, dpi=300):
    n_rows = len(csv_df)
    n_cols = len(csv_df.columns)
    fig_w = max(9.5, 1.7 * n_cols)
    fig_h = 0.42 * (n_rows + 1) + 0.15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_facecolor("white")
    ax.axis("off")

    col_labels = list(csv_df.columns)
    cell_text = csv_df.values.tolist()

    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellLoc="center", colLoc="center", loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(n_cols)))

    header_color = "#e8e8e8"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
            if row - 1 == best_idx and col == 1:  # "All-seen Accuracy (%)" column, best row
                cell.get_text().set_fontweight("bold")
        if col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02

    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _render_allseen_bar_png(display_df, best_idx, out_path, dpi=300):
    methods = display_df["Method"].tolist()
    values = display_df["_all_seen_numeric"].replace(float("-inf"), np.nan).tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#4c72b0" if i != best_idx else "#c44e52" for i in range(len(methods))]
    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="#333333", linewidth=0.6)

    for i, (bar, v) in enumerate(zip(bars, values)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9,
                 fontweight="bold" if i == best_idx else "normal")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("All-seen Accuracy (%)")
    ax.set_ylim(0, max([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], default=100) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_from_summary_csv(summary_csv_path, output_dir):
    """Convenience wrapper: reads the real final_8method_thesis_production_
    summary_table.csv this experiment writes and renders from it -- the
    values always flow from that file, never typed by hand."""
    summary_df = pd.read_csv(summary_csv_path)
    return render_final_8method_thesis_production_results(summary_df, output_dir)
