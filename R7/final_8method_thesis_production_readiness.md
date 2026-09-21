# Final 8-Method CIFAR-100 5x20 Thesis Production — Readiness Audit

**BUILD-ONLY task.** No training was run for this task. No canonical file was modified. Static
verification was performed BOTH by inspection AND by actually executing the script's config-building
logic in a truncated safe-namespace check (everything up to, but not including, `dataset =
load_dataset("cifar100")`) — this caught real bugs that inspection alone missed (Section 6 below).

## 1. Source chosen

`experiments_prepared/final_9method_5x20_performance_recovery.py` (the script that produced
job4971615) is the source of truth, per the task brief's own Section 1 instruction. This session
independently confirmed earlier (in a separate forensic task) that this script's RankExt
`restricted_mean` values reproduce job4971615's forensic summary to ~1e-14 under the same seed/config,
confirming this is the correct historical lineage. `experiments_prepared/final_8method_5x20_refined.py`
was inspected ONLY for its reporting/PNG infrastructure pattern (per the task brief's Section 1
instruction) — its RankExt task-scale calibration was explicitly NOT inherited.

## 2. Files created

- `experiments_prepared/final_8method_thesis_production.py` (new, ~9770 lines, copy-and-modify of the
  9703-line source).
- `experiments_prepared/slurm/final_8method_thesis_production.sbatch` (new launcher).
- `experiments_prepared/render_final_8method_thesis_production_results.py` (new, isolated renderer
  module — NOT a reuse of `render_final_8method_results.py`, whose `DISPLAY_ROWS` hardcodes the old
  `lam20` method identifiers that no longer exist in this run's active method set).
- `R7/final_8method_thesis_production_readiness.md` (this file).

## 3. Canonical files unchanged — MD5-verified

| File | MD5 |
|---|---|
| `experiments_prepared/final_9method_5x20_performance_recovery.py` | `048331dac367c5b940b0680f04630eed` (unchanged) |
| `experiments_prepared/final_8method_5x20_refined.py` | `4673af2628f3d47473abf0ab26bec657` (unchanged) |
| `experiments_prepared/render_final_8method_results.py` | `8f72184d8d551bd8b916337cd2487753` (unchanged) |
| `experiments_prepared/slurm/final_9method_5x20_performance_recovery.sbatch` | `a22b7b7b686a9bb98fba60388a741986` (unchanged) |
| `experiments_prepared/slurm/final_8method_5x20_refined.sbatch` | `00c2815a70ba6f3ac15a9471edf2b5e5` (unchanged) |

**PASS.** None of these files were opened for writing at any point; every new file was created via `cp`
or `Write`, then edited only at its own path.

## 4. Method list (exactly 8, verified live)

```
simple_avg
simple_avg_kd_oldseen_T2_warmup
simple_avg_dense_orth_lam1
simple_avg_dense_orth_lam1_kd_oldseen_T2_warmup
rank_extension
rank_extension_fullkd_T2_protect30
rank_extension_factor_orth_lam50
rank_extension_factor_orth_lam50_fullkd_T2_protect30
```

Confirmed via a truncated live execution (`ACTIVE_METHOD_NAMES`, `EXPECTED_METHODS`, both size 8, both
equal; `simple_avg_kd_oldseen_T4_warmup` confirmed **absent** from `ACTIVE_METHOD_MAP`; no `lam20`-named
identifier anywhere in the active set).

## 5. SimpleAvg settings

| Setting | Value | Verified |
|---|---|---|
| Head LR multiplier | 10.0 | Live: `HEAD_LR_MULTIPLIER_BY_FAMILY['simple_avg'] == 10.0`, all 4 SA arms resolve `head_lr_multiplier == 10.0` |
| LoRA LR | 5e-5 | `LR_LORA = 5e-5` (source line ~524, unchanged) |
| Rank / Alpha / Scaling | 80 / 160 / 2.0 | `LORA_R == 80 and LORA_ALPHA == 160`, asserted live |
| DenseOrth lambda | **1.0** (lowered from historical 20.0) | `DENSE_ORTH_LAMBDA == 1.0`, verified live; loss geometry (Frobenius-normalized inner product, detached-previous/differentiable-current) unchanged — only the coefficient changed |
| KD (T2 arm) temperature | 2.0 | `ACTIVE_METHOD_MAP["simple_avg_kd_oldseen_T2_warmup"]["kd_temperature"] == 2.0` |
| KD (T2 arm) weight | **1.0 — see discrepancy note below** | `kd_weight_scale=1.0` at the `add_method(...)` call site (source line ~2050-2053), `KD_WEIGHT = 1.0` globally |
| KD warmup | 1-epoch (`KD_OLDSEEN_WARMUP_EPOCHS_ARM2_ARM4`) | unchanged from source |
| Replay | none | `replay_per_class=0` unchanged throughout |

### ⚠️ Discrepancy vs. the task brief: KD weight

The task brief's Section 4 states "KD weight = 0.5" for `simple_avg_kd_oldseen_T2_warmup` and instructs
"DO NOT change KD weight in this production run." **Direct inspection of the source script shows the
actual historical, currently-active value is `kd_weight_scale=1.0` (with global `KD_WEIGHT = 1.0`),
giving an effective KD weight of 1.0, not 0.5.** Per the task brief's own explicit "do not change"
instruction, and because preserving actual historical behavior is the goal (not matching a possibly-
mistaken hand-typed number), **this production script preserves the historical `kd_weight_scale=1.0`
unchanged** — it was NOT modified to force 0.5. This is flagged here explicitly, as instructed, rather
than silently resolved either way. If a KD-weight=0.5 arm was genuinely intended, that would require a
separate, deliberate decision — not something this build silently assumed.

## 6. RankExt settings

| Setting | Value | Verified |
|---|---|---|
| Head LR multiplier | **2.0** (raised from historical 1.0) | Live: `HEAD_LR_MULTIPLIER_BY_FAMILY['rank_extension'] == 2.0`, all 4 RE arms resolve `head_lr_multiplier == 2.0` — confirmed applied uniformly across all four RankExt arms (Section 13 of the task brief) |
| Rank schedule | `[16, 32, 48, 64, 80]` | Unchanged, hard-asserted |
| FactorOrth lambda | 50.0 | `LAMBDA_ORTH == 50.0`, unchanged |
| Full KD (2 KD-bearing arms) | T=2, weight=1.0, no warmup | `kd_class_scope="full"`, `kd_weight_scale=1.0`, `kd_warmup_epochs=0.0` — all unchanged from source, verified live |
| Protect weight | 30.0 | `RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT = 30.0`; both KD-bearing RankExt arms confirmed present in `RANKEXT_PROJECTED_PROTECT_METHODS` |
| New-block warmup | disabled for the 2 KD arms, enabled for the 2 non-KD arms | Both KD-bearing arms confirmed present in `RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS`; `rank_extension_factor_orth_lam50` confirmed **absent** from both the protect and warmup-disabled sets (non-KD arm keeps warmup enabled, no protect) |
| Feature-anchor weight (non-KD arms) | 1.0 | `RANKEXT_PRETRAINED_ANCHOR_WEIGHT = 1.0`, unchanged |
| Anchor/protection mutual exclusivity | preserved | Anchor active only when `not use_kd`; protection active only when `use_kd and method in RANKEXT_PROJECTED_PROTECT_METHODS` — unchanged dispatch logic |
| Replay | none | unchanged |

## 7. Head-LR=2.0 evidence (RankExt)

Full analysis in `R7/rankext_headlr_final_compare_report.md`. Summary: a local laptop decision probe
(2 tasks, 2 epochs/task, 10/5/5 images/class, seeds 42 and 123, historical pre-task-scale-calibration
RankExt pipeline, no new calibration) found LR=2.0 beat LR=1.0 beat LR=0.5 on **both seeds**, on
validation CE, final-open accuracy, mean-restricted accuracy, and Task2-restricted accuracy, with the
improvement rate **accelerating** (not saturating) from 0.5→1.0→2.0, and no NaN/Inf/divergence/
classifier-restoration failure at any tested condition. **This is a production hypothesis informed by a
reduced-scale local probe, not proof of a globally optimal LR at full 5x20/9-epoch scale** — the local
probe's own accuracy numbers are explicitly NOT themselves a thesis benchmark result (task brief Section
25); only this full production run's real 5x20/9-epoch output is eligible as a benchmark result.

## 8. Calibration status

- **New task-scale calibration (job4972616): ABSENT.** This script's lineage
  (`final_9method_5x20_performance_recovery.py`) never contained `PerTaskScaleCalibration`,
  `RankExtCalibratedModel`, `apply_rankext_task_scale_calibration`, `fit_rankext_task_scale_calibration`,
  or `collect_rankext_logits_labels` — confirmed absent by grep before copying, and additionally
  guarded by an explicit runtime assertion (mirroring the established pattern from
  `experiments_prepared/rankext_headlr_small_probe.py`) that raises if any of those symbols exist in
  `dir()` at import time. Verified live: the safe-namespace check ran this guard block without
  exception, and `RANKEXT_TASK_SCALE_CALIBRATION_ENABLED` resolves to `False`.
- **Historical row-norm calibration (`confidence_weighted_regime_grouped`): ACTIVE, unchanged.**
  Evaluation sequence is, and remains: train → classifier restoration → historical row-norm calibration
  → final evaluation. Never: train → row-norm calibration → learned task-scale calibration.

## 9. Result-table renderer status

`experiments_prepared/render_final_8method_thesis_production_results.py` — new, isolated module,
adapted from `render_final_8method_results.py`'s established pattern (validate-then-render,
fail-loud on missing/duplicate/extra rows, T4-absence check, ≥300 DPI, no calibration row/column). New
`DISPLAY_ROWS` mapping uses this run's actual method names (lam1, not lam20) and the exact 8 display
labels from the task brief's Section 20, including `SimpleAvg + DenseOrth (λ=1)` /
`SimpleAvg + DenseOrth (λ=1) + KD`. Added an explicit "no lam20-named arm present" check as a second
line of defense beyond the missing/extra-row checks.

**Smoke-tested** (mock 8-row dataframe, scratch directory `R7/_thesis_production_render_smoke_scratch/`,
deleted immediately after): produced a valid, non-empty CSV (445 bytes) and two valid PNGs (table:
~174 KB; bar chart: ~211 KB), both correctly showing the λ=1 display labels and the em-dash for
SimpleAvg's structurally-undefined Forgetting cells. Three negative-path smoke tests confirmed the
validator raises (does not silently render) on: a missing arm, a T4-arm-present summary, and a
lam20-named-arm-present summary. No mock output survives.

Integration into the main script: wired into the reporting section (mirroring
`final_8method_5x20_refined.py`'s own end-of-run hook pattern, including the `sys.path` self-injection
for import robustness regardless of invocation CWD), called once, immediately after the job4971615
comparison table is written, passing the real `final9_summary_df`. Expected real output:
`R7/final_8method_thesis_results_table.csv`, `R7/final_8method_thesis_results_table.png`,
`R7/final_8method_thesis_allseen_accuracy.png` — produced automatically the next time this script
completes a real run.

## 10. SLURM launcher status

`experiments_prepared/slurm/final_8method_thesis_production.sbatch` — new, distinct
job-name/output/error, `bash -n` syntax-checked clean, points at
`experiments_prepared/final_8method_thesis_production.py`, 16-hour walltime (vs. job4971615's 18h,
scaled down for one fewer SimpleAvg arm).

## 11. Exact config-diff audit vs. job4971615 (task brief Section 18)

| # | Change | Verified |
|---|---|---|
| 1 | T4 arm removed | ✅ `simple_avg_kd_oldseen_T4_warmup` confirmed absent from `ACTIVE_METHOD_MAP`; `METHODS_TO_RUN` flag flipped `False`; loud standing guard assertion added |
| 2 | SimpleAvg DenseOrth lambda 20 → 1 | ✅ `DENSE_ORTH_LAMBDA == 1.0`, both DenseOrth method identifiers renamed lam20→lam1 everywhere (active code only; historical reference-value rows describing job4969059/job4970580's genuine lambda=20 runs correctly left unchanged) |
| 3 | SimpleAvg head LR remains ×10 | ✅ unchanged, verified live |
| 4 | KD remains unchanged | ✅ historical `kd_weight_scale=1.0`/T=2/old-seen scope preserved exactly — see the discrepancy note in Section 5 above regarding the task brief's own restated (incorrect) "0.5" figure |
| 5 | RankExt head LR 1 → 2 | ✅ `HEAD_LR_MULTIPLIER_BY_FAMILY['rank_extension'] == 2.0`, applied to all 4 RankExt arms, verified live |
| 6 | New learned task-scale calibration NOT included | ✅ confirmed absent by construction + runtime guard |
| 7 | Historical row-norm calibration retained | ✅ unchanged dispatch logic, `confidence_weighted_regime_grouped` mode preserved |

**No other scientific changes were made.** Everything else (LoRA targets, optimizer, scheduler, batch
size, epoch count, seed, class order/splits, replay=off, FactorOrth lambda=50, protect weight=30,
feature-anchor weight=1.0, new-block-warmup gating) is byte-identical to the source script's logic.

## 12. Additional reporting added (task brief Sections 19-23)

- **SimpleAvg pre/post-historical-calibration diagnostics** (Section 19): a new, fully isolated
  accumulator (`simpleavg_calibration_diagnostic_rows`) populated only from the real, active evaluation
  path (`evaluate_arm_with_merge()`'s `merge_method=="arithmetic"` branch — NOT the preserved-but-unused
  `run_simple_avg_variant()`), via a new side-effect-free evaluation helper
  (`evaluate_all_seen_accuracy_only`) that does **not** touch `all_results`/`method_summary_rows` or any
  accumulator the main summary table reads from. Written to its own dedicated CSV
  (`final_8method_thesis_production_simpleavg_calibration_diagnostics.csv`), never merged into or read
  by the main 8-row summary table. No new benchmark method row is created.
- **DenseOrth/CE ratio diagnostics** (Section 8): already present, unmodified, in the existing
  `IndependentLoraOrthTrainer.compute_loss()` per-batch logging (`dense_orth_mean`,
  `lambda_orth_times_loss`, `orth_ratio_abs_weighted_over_ce`/`weighted_orth_over_CE`) — no new
  instrumentation needed; no expensive extra-backward-pass gradient-ratio diagnostic was added (none
  existed cheaply to reuse, and the task brief explicitly says not to make this mandatory if it would
  slow/alter production training).
- **Comparison against job4971615** (Section 23): a new, separate comparison table
  (`final_8method_thesis_production_vs_job4971615.csv`), explicitly distinct from the pre-existing
  job4970580/R7-Exp1 "CONFOUNDED" comparison table (which compares against the WRONG historical baseline
  for this task's purposes and is left untouched/unrenamed for provenance). Each row is labeled with
  exactly which controlled change it tests — `"lambda 20 -> 1"` for the two SimpleAvg DenseOrth rows,
  `"head LR 1 -> 2"` for all four RankExt rows — per the task brief's explicit "do not conflate"
  instruction.

## 13. Bugs found and fixed via live execution (not caught by static reading alone)

A truncated safe-namespace check (executing the script's actual config-building/assertion code, up to
but excluding `dataset = load_dataset("cifar100")` — no CIFAR-100 download, no CLIP model load, no
GPU/training work) found and this task fixed **three** stale hardcoded-count assertions inherited from
the 9-method source that would have crashed the real run immediately, before any training began:

1. `assert sum(... family == "simple_avg") == 5` → fixed to `== 4`.
2. `assert len(ACTIVE_METHOD_NAMES) == 9` → fixed to `== 8`.
3. `assert len(_sa_arms) == 5 and len(_re_arms) == 4 and len(ACTIVE_METHOD_NAMES) == 9` (and the paired
   `_expected_final_configs == 9`) → fixed to `4`/`4`/`8` and `8` respectively.

Also fixed two non-crashing but misleading artifacts found during the same pass: `RUN_NAME_BASE` had
not been changed from the source's own name (`cifar100_5x20_final_9method_performance_recovery_seed42`)
— this would have made this production run's checkpoints/output directory indistinguishable in name
from a job4971615-lineage run, though not an actual collision risk since the source script itself is
never run under that exact name; fixed to
`cifar100_5x20_final_8method_thesis_production_seed{SEED}`. A cosmetic `"SCRIPT UNDER TEST:
final_9method_5x20_performance_recovery.py"` print was also corrected to reference this script's own
filename. After all three assertion fixes, the full truncated safe-namespace check ran end-to-end
without exception, and explicit live verification confirmed every Section 26 config value listed below.

A handful of purely cosmetic comment/print labels elsewhere in the file (e.g. "for ALL 9 arms" section
headers in the reporting code, a historical header comment describing the original 9-method design
brief) still say "9" — these are non-functional prose, do not affect behavior, and were left unchanged,
consistent with this codebase's established convention of preserving historical provenance comments
verbatim elsewhere in the same file lineage.

## 14. Static/safety audit (task brief Section 26) — all verified live, not just by inspection

| Check | Result |
|---|---|
| Exactly 8 methods, no T4, no extra LR arms, no lam20 SA arm | ✅ verified live |
| SimpleAvg: head LR=10, LoRA LR=5e-5, rank=80, alpha=160, DenseOrth lambda=1 | ✅ verified live |
| SimpleAvg: KD weight/T | KD T=2.0 confirmed; KD weight is 1.0 (historical), not 0.5 as the task brief stated — see Section 5 discrepancy note |
| RankExt: head LR=2 (all 4 arms), rank schedule unchanged, FactorOrth lambda=50, full KD weight=1 T=2, protect30 | ✅ verified live |
| New task-scale calibration absent; old row-norm calibration active | ✅ verified live (guard assertion passes) |
| Seed 42, 5x20 protocol, full epoch count (9/9), full dataset (no per-class subsampling), no replay | ✅ verified live — this is the full production dataset, not a micro-probe |
| Canonical files untouched | ✅ MD5-verified before/after |
| `python -m py_compile` | ✅ PASS, both the main script and the new renderer module |
| `bash -n` on the new launcher | ✅ PASS |

## 15. Exact submission command (NOT run in this task)

```
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
sbatch experiments_prepared/slurm/final_8method_thesis_production.sbatch
```

---

# FINAL TERMINAL SUMMARY

**FINAL 8-METHOD THESIS PRODUCTION PREPARATION**

**METHODS**
Count: **8**
T4 present: **NO**
SA lambda20 present: **NO**
Exact list: `simple_avg`, `simple_avg_kd_oldseen_T2_warmup`, `simple_avg_dense_orth_lam1`,
`simple_avg_dense_orth_lam1_kd_oldseen_T2_warmup`, `rank_extension`,
`rank_extension_fullkd_T2_protect30`, `rank_extension_factor_orth_lam50`,
`rank_extension_factor_orth_lam50_fullkd_T2_protect30`

**SIMPLEAVG**
Head LR: **10.0**
LoRA LR: **5e-5**
Rank: **80**
Alpha: **160**
KD T: **2.0**
KD weight: **1.0 (historical value — NOT 0.5 as the task brief stated; preserved unchanged per the
brief's own "do not change KD weight" instruction — see Section 5 discrepancy note above)**
DenseOrth lambda: **1.0** (lowered from historical 20.0)
Replay: **none**

**RANKEXT**
Head LR: **2.0** (raised from historical 1.0, applied to all 4 arms)
Rank schedule: **[16, 32, 48, 64, 80]** (unchanged)
Full KD T: **2.0**
KD weight: **1.0**
Protect: **30.0**
FactorOrth lambda: **50.0** (unchanged)
New task-scale calibration: **ABSENT** (runtime-asserted)
Historical row-norm calibration: **ACTIVE**
Replay: **none**

**CONTROLLED CHANGES VS JOB4971615**
SA DenseOrth lambda: **20 → 1**
RankExt head LR: **1 → 2**
T4: **removed**
Other scientific changes: **NONE**

**REPORTING**
Main result rows: **8** (thesis display names: SimpleAvg, SimpleAvg + KD, SimpleAvg + DenseOrth (λ=1),
SimpleAvg + DenseOrth (λ=1) + KD, RankExt, RankExt + KD + Protect, RankExt + FactorOrth,
RankExt + FactorOrth + KD + Protect)
Raw/pre-calibration SA diagnostics: **YES, isolated, separate CSV, never merged into main table**
PNG table renderer: **NEW isolated module, smoke-tested with mock data + 3 negative-path checks, all PASS**
Comparison vs job4971615: **YES, separate table, lambda-change and head-LR-change rows explicitly
labeled, not conflated**

**STATIC CHECKS**
Python compile: **PASS** (main script + renderer module)
SLURM syntax: **PASS**
8-method assertion: **PASS** (verified live via truncated safe-namespace execution; 3 stale
hardcoded-count bugs found and fixed in this process — see Section 13)
Canonical files unchanged: **YES** (MD5-verified)

**FILES**
Experiment: `experiments_prepared/final_8method_thesis_production.py`
Launcher: `experiments_prepared/slurm/final_8method_thesis_production.sbatch`
Readiness report: `R7/final_8method_thesis_production_readiness.md` (this file)
Renderer/output integration: `experiments_prepared/render_final_8method_thesis_production_results.py`
(new, isolated, wired into the reporting section)

**FULL TRAINING SUBMITTED: NO**
**READY TO SUBMIT: YES**
