# Final 8-method table audit

This package contains tables only. No thesis prose, chapter text, training outputs, checkpoints, logits, or canonical CIFAR results were edited.

## Table checks

- Eight CUB display rows: PASS (exactly 8).
- CUB row order: PASS (matches the requested eight-row order).
- Duplicate display methods: PASS (none).
- Final method name: PASS (`RankExt Normalized KD+FactorOrth`).
- Final CUB All-seen: PASS (63.428%).
- Final CUB Restricted: PASS (88.385%).
- Best CUB All-seen: PASS (SimpleAvg, 69.468%).
- Final gap: PASS (69.468 − 63.428 = 6.040 pp).
- SimpleAvg Forgetting: PASS (reported as `—`; not fabricated).
- Canonical CIFAR values: PASS (validated against the job4971615 table).

## CUB row sources

| Row | Display method | Source |
|---:|---|---|
| 1 | SimpleAvg | CUB final_8method_summary_table.csv: simple_avg |
| 2 | SimpleAvg + KD | CUB final_8method_summary_table.csv: simple_avg_kd_oldseen_T2_warmup |
| 3 | SimpleAvg + DenseOrth | CUB final_8method_summary_table.csv: simple_avg_dense_orth_lam20 |
| 4 | SimpleAvg + DenseOrth + KD | CUB final_8method_summary_table.csv: simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup |
| 5 | RankExt | CUB final_8method_summary_table.csv: rank_extension |
| 6 | RankExt KD+Protect | CUB final_8method_summary_table.csv: rank_extension_fullkd_T2_protect30 |
| 7 | RankExt Normalized FactorOrth | CUB final_8method_summary_table.csv: rank_extension_factor_orth_normalized_lam50 |
| 8 | RankExt Normalized KD+FactorOrth | CUB task-block-scale calibration report for All-seen/Restricted; raw final_8method_summary_table.csv for BWT/Forgetting |

The eighth CUB row intentionally combines the final selected task-block-scale All-seen/Restricted result with the raw canonical method's BWT/Forgetting trajectory metrics. The calibration evidence is validation-selected and final-evaluation-only; no calibrated BWT/Forgetting was claimed.

## CIFAR row sources

All eight CIFAR rows come from `R7/chapter5_main_benchmark/tables/cifar100_main_8method_table.csv`, the canonical job4971615 final 8-method table. The cross-dataset table preserves canonical CIFAR method names in its dedicated column. In particular, CIFAR row 8 remains the canonical Combined result and is not renamed as normalized FactorOrth.

## Source integrity

The generation process opened source files read-only and wrote only files under `R8/final_tables/`. Underlying raw CUB artifacts and canonical CIFAR results were untouched.

| Source file | SHA-256 recorded after audit |
|---|---|
| `R7\cub200_5x40_final_8method_improved_rankext_seed42_ep9_EPOCH9_MAIN_20260923_171203\tables\final_8method_summary_table.csv` | `30da82191fa81592aad7756d760306e76bb8bbb22eeaf6cd98cf2a8bbeee8e12` |
| `R8\performance_improvement_research\cub_job4983091_normalized_combined_hierarchical_calibration.md` | `736dfc5e5f0de67ee6af2d5ad437bc62694dbbaac9caea46e1cf02dad701c264` |
| `R7\chapter5_main_benchmark\tables\cifar100_main_8method_table.csv` | `aa6837377c7e13f8e91b53997e4c34104606aba73e21ecee11c51e1e4b376128` |

## Generated files

- `cub200_final_8method_results.csv` and `.md`
- `cub200_vs_cifar100_final_8method_comparison.csv` and `.md`
- `cub200_vs_cifar100_all_seen_delta.csv` and `.md`
- `final_8method_table_audit.md`
- `generate_final_8method_tables.py`
