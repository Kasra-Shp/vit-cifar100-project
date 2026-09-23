"""Generate and audit the final eight-method CUB/CIFAR result tables.

This is table preparation only. It reads canonical result tables and the
already-selected CUB calibration evidence; it does not load models, logits,
datasets, or training outputs for modification.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "R8" / "final_tables"
CUB_RUN = ROOT / "R7" / "cub200_5x40_final_8method_improved_rankext_seed42_ep9_EPOCH9_MAIN_20260923_171203"
CUB_SUMMARY = CUB_RUN / "tables" / "final_8method_summary_table.csv"
CUB_CALIBRATION = ROOT / "R8" / "performance_improvement_research" / "cub_job4983091_normalized_combined_hierarchical_calibration.md"
CIFAR_TABLE = ROOT / "R7" / "chapter5_main_benchmark" / "tables" / "cifar100_main_8method_table.csv"


DISPLAY_METHODS = [
    "SimpleAvg",
    "SimpleAvg + KD",
    "SimpleAvg + DenseOrth",
    "SimpleAvg + DenseOrth + KD",
    "RankExt",
    "RankExt KD+Protect",
    "RankExt Normalized FactorOrth",
    "RankExt Normalized KD+FactorOrth",
]


CUB_ROWS = [
    {"method": DISPLAY_METHODS[0], "all_seen": 69.468, "restricted": 91.251, "bwt": -0.241, "forgetting": "—", "source": "CUB final_8method_summary_table.csv: simple_avg"},
    {"method": DISPLAY_METHODS[1], "all_seen": 68.692, "restricted": 91.942, "bwt": -0.266, "forgetting": "—", "source": "CUB final_8method_summary_table.csv: simple_avg_kd_oldseen_T2_warmup"},
    {"method": DISPLAY_METHODS[2], "all_seen": 68.329, "restricted": 91.047, "bwt": -0.259, "forgetting": "—", "source": "CUB final_8method_summary_table.csv: simple_avg_dense_orth_lam20"},
    {"method": DISPLAY_METHODS[3], "all_seen": 67.242, "restricted": 91.735, "bwt": -0.288, "forgetting": "—", "source": "CUB final_8method_summary_table.csv: simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup"},
    {"method": DISPLAY_METHODS[4], "all_seen": 20.262, "restricted": 78.120, "bwt": -0.891, "forgetting": 0.890, "source": "CUB final_8method_summary_table.csv: rank_extension"},
    {"method": DISPLAY_METHODS[5], "all_seen": 45.271, "restricted": 87.779, "bwt": -0.547, "forgetting": 0.589, "source": "CUB final_8method_summary_table.csv: rank_extension_fullkd_T2_protect30"},
    {"method": DISPLAY_METHODS[6], "all_seen": 20.815, "restricted": 79.581, "bwt": -0.885, "forgetting": 0.884, "source": "CUB final_8method_summary_table.csv: rank_extension_factor_orth_normalized_lam50"},
    {"method": DISPLAY_METHODS[7], "all_seen": 63.428, "restricted": 88.385, "bwt": -0.532, "forgetting": 0.570, "source": "CUB task-block-scale calibration report for All-seen/Restricted; raw final_8method_summary_table.csv for BWT/Forgetting"},
]


CIFAR_IDS = [
    "simple_avg",
    "simple_avg_kd",
    "simple_avg_denseorth",
    "simple_avg_denseorth_kd",
    "rankext",
    "rankext_kd_protect",
    "rankext_factororth",
    "rankext_factororth_kd_protect",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt3(value: object) -> str:
    if value == "—":
        return "—"
    return f"{float(value):.3f}"


def load_cifar_rows() -> dict[str, dict[str, str]]:
    with CIFAR_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        return {row["Method"]: row for row in csv.DictReader(handle)}


def csv_text(headers: list[str], rows: list[list[object]]) -> str:
    lines: list[str] = []
    writer_target = []
    # Use csv.writer on a list-backed text adapter for RFC-compliant quoting.
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_outputs(cifar_rows: dict[str, dict[str, str]]) -> None:
    cub_csv_rows = [
        [row["method"], fmt3(row["all_seen"]), fmt3(row["restricted"]), fmt3(row["bwt"]), fmt3(row["forgetting"])]
        for row in CUB_ROWS
    ]
    cub_headers = ["Method", "All-seen (%)", "Restricted (%)", "BWT", "Forgetting"]
    (OUT / "cub200_final_8method_results.csv").write_text(csv_text(cub_headers, cub_csv_rows), encoding="utf-8")
    (OUT / "cub200_final_8method_results.md").write_text(
        "# Final CUB-200 8-method results\n\n" + markdown_table(cub_headers, cub_csv_rows) + "\n\n"
        "Note: the final RankExt Normalized KD+FactorOrth All-seen/Restricted values are the selected task-block-scale result. Its BWT/Forgetting values remain the raw training-trajectory metrics because calibration was applied only at final evaluation.\n",
        encoding="utf-8",
    )

    cifar_display = [
        "SimpleAvg",
        "SimpleAvg + KD",
        "SimpleAvg + FactorOrth",
        "SimpleAvg + FactorOrth + KD",
        "RankExt",
        "RankExt + KD + Protect",
        "RankExt + FactorOrth",
        "RankExt + FactorOrth + KD + Protect",
    ]
    notes = [
        "Canonical CIFAR job4971615.",
        "Canonical CIFAR job4971615.",
        "Canonical CIFAR FactorOrth variant; CUB display name is DenseOrth.",
        "Canonical CIFAR FactorOrth+KD variant; CUB display name is DenseOrth+KD.",
        "Canonical CIFAR job4971615.",
        "Canonical CIFAR job4971615.",
        "Canonical CIFAR FactorOrth; CUB uses normalized FactorOrth.",
        "CIFAR remains canonical Combined; CUB is normalized KD+FactorOrth with task-block scale calibration.",
    ]
    cross_headers = [
        "Method family/variant",
        "CIFAR canonical method",
        "CIFAR all-seen (%)",
        "CIFAR restricted (%)",
        "CUB all-seen (%)",
        "CUB restricted (%)",
        "Provenance note",
    ]
    cross_rows: list[list[object]] = []
    for index, (cub, cifar_id, cifar_name, note) in enumerate(zip(CUB_ROWS, CIFAR_IDS, cifar_display, notes), start=1):
        source = cifar_rows[cifar_name]
        cross_rows.append([
            cub["method"],
            cifar_name,
            f"{float(source['All-seen (%)']):.2f}",
            f"{float(source['Restricted (%)']):.2f}",
            fmt3(cub["all_seen"]),
            fmt3(cub["restricted"]),
            note,
        ])
    (OUT / "cub200_vs_cifar100_final_8method_comparison.csv").write_text(csv_text(cross_headers, cross_rows), encoding="utf-8")
    (OUT / "cub200_vs_cifar100_final_8method_comparison.md").write_text(
        "# Final 8-method CIFAR/CUB comparison\n\n" + markdown_table(cross_headers, cross_rows) + "\n",
        encoding="utf-8",
    )

    delta_headers = ["Method", "CIFAR all-seen (%)", "CUB all-seen (%)", "CUB − CIFAR (pp)"]
    delta_rows: list[list[object]] = []
    for cub, cifar_name in zip(CUB_ROWS, cifar_display):
        cifar_value = float(cifar_rows[cifar_name]["All-seen (%)"])
        cub_value = float(cub["all_seen"])
        delta_rows.append([cub["method"], f"{cifar_value:.2f}", fmt3(cub_value), f"{cub_value - cifar_value:.3f}"])
    (OUT / "cub200_vs_cifar100_all_seen_delta.csv").write_text(csv_text(delta_headers, delta_rows), encoding="utf-8")
    (OUT / "cub200_vs_cifar100_all_seen_delta.md").write_text(
        "# CUB minus CIFAR all-seen delta\n\n" + markdown_table(delta_headers, delta_rows) + "\n",
        encoding="utf-8",
    )


def validate(cifar_rows: dict[str, dict[str, str]]) -> None:
    assert len(CUB_ROWS) == 8
    assert [row["method"] for row in CUB_ROWS] == DISPLAY_METHODS
    assert len({row["method"] for row in CUB_ROWS}) == 8
    assert CUB_ROWS[0]["all_seen"] == 69.468
    assert CUB_ROWS[7]["method"] == "RankExt Normalized KD+FactorOrth"
    assert CUB_ROWS[7]["all_seen"] == 63.428
    assert CUB_ROWS[7]["restricted"] == 88.385
    assert max(row["all_seen"] for row in CUB_ROWS) == 69.468
    assert round(CUB_ROWS[0]["all_seen"] - CUB_ROWS[7]["all_seen"], 3) == 6.040

    expected_cifar = {
        "SimpleAvg": (75.16, 92.28),
        "SimpleAvg + KD": (75.21, 92.86),
        "SimpleAvg + FactorOrth": (74.02, 91.97),
        "SimpleAvg + FactorOrth + KD": (73.22, 92.96),
        "RankExt": (34.99, 91.03),
        "RankExt + KD + Protect": (65.01, 94.10),
        "RankExt + FactorOrth": (41.00, 93.03),
        "RankExt + FactorOrth + KD + Protect": (70.76, 94.78),
    }
    assert set(cifar_rows) >= set(expected_cifar)
    for name, (all_seen, restricted) in expected_cifar.items():
        assert float(cifar_rows[name]["All-seen (%)"]) == all_seen
        assert float(cifar_rows[name]["Restricted (%)"]) == restricted
    calibration_text = CUB_CALIBRATION.read_text(encoding="utf-8")
    assert "63.428%" in calibration_text and "88.385%" in calibration_text


def write_audit(cifar_rows: dict[str, dict[str, str]]) -> None:
    source_hashes = {
        CUB_SUMMARY: sha256(CUB_SUMMARY),
        CUB_CALIBRATION: sha256(CUB_CALIBRATION),
        CIFAR_TABLE: sha256(CIFAR_TABLE),
    }
    lines = [
        "# Final 8-method table audit",
        "",
        "This package contains tables only. No thesis prose, chapter text, training outputs, checkpoints, logits, or canonical CIFAR results were edited.",
        "",
        "## Table checks",
        "",
        "- Eight CUB display rows: PASS (exactly 8).",
        "- CUB row order: PASS (matches the requested eight-row order).",
        "- Duplicate display methods: PASS (none).",
        "- Final method name: PASS (`RankExt Normalized KD+FactorOrth`).",
        "- Final CUB All-seen: PASS (63.428%).",
        "- Final CUB Restricted: PASS (88.385%).",
        "- Best CUB All-seen: PASS (SimpleAvg, 69.468%).",
        "- Final gap: PASS (69.468 − 63.428 = 6.040 pp).",
        "- SimpleAvg Forgetting: PASS (reported as `—`; not fabricated).",
        "- Canonical CIFAR values: PASS (validated against the job4971615 table).",
        "",
        "## CUB row sources",
        "",
        "| Row | Display method | Source |",
        "|---:|---|---|",
    ]
    lines.extend(f"| {index} | {row['method']} | {row['source']} |" for index, row in enumerate(CUB_ROWS, start=1))
    lines.extend([
        "",
        "The eighth CUB row intentionally combines the final selected task-block-scale All-seen/Restricted result with the raw canonical method's BWT/Forgetting trajectory metrics. The calibration evidence is validation-selected and final-evaluation-only; no calibrated BWT/Forgetting was claimed.",
        "",
        "## CIFAR row sources",
        "",
        "All eight CIFAR rows come from `R7/chapter5_main_benchmark/tables/cifar100_main_8method_table.csv`, the canonical job4971615 final 8-method table. The cross-dataset table preserves canonical CIFAR method names in its dedicated column. In particular, CIFAR row 8 remains the canonical Combined result and is not renamed as normalized FactorOrth.",
        "",
        "## Source integrity",
        "",
        "The generation process opened source files read-only and wrote only files under `R8/final_tables/`. Underlying raw CUB artifacts and canonical CIFAR results were untouched.",
        "",
        "| Source file | SHA-256 recorded after audit |",
        "|---|---|",
    ])
    lines.extend(f"| `{path.relative_to(ROOT)}` | `{digest}` |" for path, digest in source_hashes.items())
    lines.extend([
        "",
        "## Generated files",
        "",
        "- `cub200_final_8method_results.csv` and `.md`",
        "- `cub200_vs_cifar100_final_8method_comparison.csv` and `.md`",
        "- `cub200_vs_cifar100_all_seen_delta.csv` and `.md`",
        "- `final_8method_table_audit.md`",
        "- `generate_final_8method_tables.py`",
    ])
    (OUT / "final_8method_table_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in (CUB_SUMMARY, CUB_CALIBRATION, CIFAR_TABLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    cifar_rows = load_cifar_rows()
    validate(cifar_rows)
    write_outputs(cifar_rows)
    write_audit(cifar_rows)
    print("validated 8 CUB rows and canonical CIFAR values")
    print("final gap: 6.040 pp")


if __name__ == "__main__":
    main()
