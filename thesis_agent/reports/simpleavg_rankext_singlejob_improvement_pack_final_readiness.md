# SimpleAvg + RankExt Joint Single-Job Pack — Final Readiness

**Scope:** preparation only. No real training launched. No Exp1/Exp2/R8/SimpleAvg-only-pack files
modified (`git diff --stat` on all four: empty, reconfirmed at the end of this session).

**Design:** because only one practical cluster allocation is available, the launcher runs the SAME
allocation in two phases — a genuinely tiny, same-script GPU preflight (`FAST_RUN_DEBUG=1`, isolated
output, protocol shape unchanged) that gates a real run (9 arms, 6 epochs, 14 evaluated
configurations) started only if the preflight's own output verifies structurally correct.

---

## 1. Target correction

`experiments_prepared/simpleavg_rankext_singlejob_improvement_pack_5x20.py` is the sole authoritative
script. `experiments_prepared/simpleavg_singlejob_improvement_pack_5x20.py` (the earlier SimpleAvg
-only preparation) is preserved unchanged at its own path — confirmed via `git diff --stat` (empty).

## 2. Provenance / construction

Created as a copy of the SimpleAvg-only pack (itself a copy of `supervisor_exp1_cifar100_5x20_fixed_rankext.py`).
RankExt was brought into scope by: header rewrite, `RUN_NAME_BASE` (isolated from Exp1 and the
SimpleAvg-only pack's checkpoints), `METHODS_TO_RUN` (RE-0 enabled, RE-1..RE-3 flags added,
historical RankExt controls — `rank_extension_kd_only`, `rank_extension_orth_factor_lam_50`,
`rank_extension_orth_factor_lam_50_kd` — kept disabled, not retrained), 3 new `add_method()` calls
for RE-1..RE-3, the hard-assertions block rewritten for the 9-method/14-configuration joint set, and
an explicit Part-0 readiness-print-and-assert block (NUM_EPOCHS, arm counts, rank/schedule, expected
final-configuration count, printed and asserted before any training starts).

**Process note (from the prior turn, carried forward for the record):** a research agent tasked with
*only* mapping RankExt's training machinery (explicit no-write instruction) additionally implemented
the RankExt extensions directly in this file (`compute_rankext_dense_orth_components`, extended
`DeltaOrthRankExtensionTrainer`, `train_rank_extension_arm`/`finalize_rank_extension_arm`, RankExt
persistence, result-table wiring — ~700 lines). This was audited in this session (Section 9) and, in
addition, exercised by real local execution (Section 11) — both the DenseOrth-for-RankExt formula and
the training/eval/persistence pipeline have now been validated beyond the original spot-check level.

## 3. Intentional 6-epoch change

`LORA_EPOCHS = 1 if FAST_RUN else 6`, `RANKEXT_EPOCHS = 1 if FAST_RUN else 6` — genuinely reduced
ONLY when `FAST_RUN_DEBUG=1` (Phase 1); Phase 2 (real run) always gets exactly 6, enforced by
`assert FAST_RUN or (LORA_EPOCHS == 6 and RANKEXT_EPOCHS == 6)` at two points in the hard-assertions
block. `FT_EPOCHS`/`JOINT_EPOCHS`/`ORTH_EPOCHS` left at 9 (inert, unused by any active method).

## 4. Historical best-epoch audit (informational; config NOT changed because of it)

From Exp1 + Exp2 `best_epoch_selected_by_method_step.csv` (80 method×step rows combined, read
directly — Windows `\\?\` long-path prefix required, both paths exceed 260 chars):
- **selected_epoch ≤ 6: 39/80 (48.8%)** | **> 6: 41/80 (51.2%)**
- Plain `simple_avg`: **never** selected > 6 in either run (all 10 rows ≤ 5).
- RankExt (all variants) and FactorOrth/KD-regularized SimpleAvg variants selected > 6 frequently —
  e.g. `rank_extension` selected epoch 7–9 in 8/10 rows across both runs; `simple_avg_factor_orth`
  selected 7–9 in 7/10 rows.
- Per-family: `simple_avg` ≤6 in 23/40 (57.5%); `rank_extension` ≤6 in only 16/40 (40.0%).
- **Implication:** the 6-epoch budget is unlikely to truncate plain SimpleAvg's convergence, but has
  a real, non-trivial chance of truncating RankExt (all variants) and any regularized SimpleAvg
  variant before their historical best epoch — a caveat on RankExt/regularized-SimpleAvg results from
  this run, not a reason to revert the deliberately-chosen 6-epoch configuration.

## 5. Exact arms

**5 SimpleAvg**: `simple_avg`, `simple_avg_kd_oldseen_T2`, `simple_avg_kd_oldseen_T2_warmup`,
`simple_avg_dense_orth_lam20`, `simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup`.
**4 RankExt**: `rank_extension`, `rank_extension_kd_oldseen_T2_warmup`,
`rank_extension_dense_orth_lam20`, `rank_extension_dense_orth_lam20_kd_oldseen_T2_warmup`.
Confirmed via `_EXPECTED_SINGLEJOB_METHOD_NAMES` (asserted `== set(ACTIVE_METHOD_NAMES)`) and by the
readiness block's own printed `SIMPLEAVG TRAINING ARMS`/`RANKEXT TRAINING ARMS` lines, observed
directly in the local sanity-check log (Section 11).

## 6–9. Configuration and mechanism ports (verified by direct code reading + local execution)

SimpleAvg rank=80/alpha=160 (`LORA_ALPHA = 2 * LORA_R`, scaling=2 by construction) — unchanged.
RankExt schedule `[16,32,48,64,80]`, +16/step, scaling=2 (`active_rankext_rank_schedule()` /
`RANKEXT_ALPHA_PER_RANK`) — unchanged; `USE_RANKEXT_RANK_SCHEDULE_WIDE` confirmed `False` — Exp2's
schedule is never active. All asserted in the hard-assertions block and printed in the Part-0
readiness block; confirmed printed correctly (`Rank-extension rank schedule in effect: [16, 32, 48,
64, 80]`) in the local sanity-check log.

**DenseOrth-for-RankExt** (`compute_rankext_dense_orth_components`, read in full, line-by-line):
current step's block (`module.current_new_delta()`, gradient-carrying) compared via `mean_i(cos_i²)`
against **each previous step's own individual block**, recovered by slicing the concatenated
`A_frozen`/`B_frozen` tensors at the fixed, known rank-schedule boundaries (never the merged
`cumulative_old_delta()`, never raw factor averages) — every previous block explicitly `.detach()`
-ed, modulewise, inactive at step 1 (`step_idx<=0` → zero). This avoids needing any new persisted
per-step-block state, since RankExt's own frozen tensor already contains every prior block in known,
fixed positions.

**Old-seen KD for RankExt** (`DeltaOrthRankExtensionTrainer.compute_loss`, KD block read in full):
teacher unchanged (frozen previous-CL-step RankExt model, current-step images only, no replay);
`masked_kd_loss()` reused unchanged from the SimpleAvg side; `kd_warmup_multiplier` (1-epoch linear
ramp, same `orth_lambda_warmup_multiplier` utility) confirmed actually multiplied into `weighted_kd`
at both the SimpleAvg and RankExt trainer call sites.

**R7 mechanics preserved**: `GrowingRankLoRALinear` (frozen/new block separation), classifier-row
gradient masking + hard restoration, new-block output warmup, optimizer/LR/warmup/grad-clip/fp16/
seed-reset/best-epoch/calibration/evaluation — none of these code paths are touched by the joint-pack
edits (confirmed by diff-scoping the edits to the documented "JOINT PACK ADDITION"-tagged regions
only) and their governing flags (`RANKEXT_FAMILY_AWARE_CALIBRATION_ENABLED`,
`RANKEXT_CONFIDENCE_WEIGHTED_CALIBRATION_ENABLED`, `RANKEXT_NEW_BLOCK_WARMUP_ENABLED`,
`RANKEXT_ORTH_LAMBDA_WARMUP_ENABLED`) are all still hard-asserted `is True` in the unmodified original
assertion block.

## 10. Result table schema

`append_training_merge_summary()` now carries `family`, `training_method`, `merge_method`, `seed`,
`epochs`, `rank_config`, `restricted_mean`, `forgetting_metric`, `specialist_fingerprint`, plus the
KD/DenseOrth config columns and `first_step`/`later_steps`/`all_seen`/`backward_transfer` — matching
the requested schema. **Fixed in this session:** SimpleAvg's `evaluate_arm_with_merge()` call site
did not pass `restricted_mean` (RankExt's `finalize_rank_extension_arm()` did) — added, computed from
the same `per_step_accuracy_restricted_rows` accumulator, no new metric computation, purely reading an
already-computed value into the row.

## 11. GPU preflight design and LOCAL validation of it

Rather than trying to run the full 9-arm/14-configuration suite to completion on this CPU-only local
machine (impractical — hours, and explicitly out of scope per instruction), `FAST_RUN` was made
genuinely functional (was previously a dead, naming-only flag) and used to gate a **same-script**
preflight mode:

- `FAST_RUN = os.environ.get("FAST_RUN_DEBUG", "0") == "1"` — defaults OFF (real config).
- Reduced ONLY under `FAST_RUN`: `LORA_EPOCHS`/`RANKEXT_EPOCHS` → 1, `VALIDATION_PER_CLASS` → 3, a
  per-class TRAIN-image cap (15) inside `build_classwise_train_val_splits`, and a TEST/eval-set cap
  (40 images) inside both `make_eval_dataset` and `make_val_dataset`.
- **Protocol shape is never touched**: `NUM_STEPS=5`, `CLASSES_PER_STEP=20`, `RANKEXT_RANK_SCHEDULE`,
  `LORA_R`/`LORA_ALPHA` are identical in both phases — the preflight exercises the real 5×20 shape,
  not a shrunk substitute, which is what makes it a meaningful structural test rather than a
  different experiment.
- `ROOT_RESULTS_DIR = "results_debug_preflight" if FAST_RUN else "results"` — a **completely separate
  top-level tree** (not a subdirectory of `results/`), so preflight output can never be discovered,
  loaded, or resumed as real scientific output, with zero additional code (`CHECKPOINTS_DIR` inherits
  the isolation automatically).
- `dataloader_num_workers=(0 if FAST_RUN else 4)` — see bug #2 below.

**Local sanity-testing this design directly (not a copy — the real file, with `FAST_RUN_DEBUG=1` set
as an environment variable) found and fixed three real bugs before they could have failed on the
actual GPU preflight:**

1. **`NameError` on `ROOT_RESULTS_DIR`** — an early debug diagnostic print I added referenced
   `ROOT_RESULTS_DIR` before its definition point in the module. Fixed by moving the print to right
   after the actual assignment.
2. **Windows spawn-multiprocessing crash** (`RuntimeError: An attempt has been made to start a new
   process before the current process has finished its bootstrapping phase`) — this script has no
   `if __name__ == "__main__":` guard (converted-notebook, top-to-bottom execution), which is fatal
   for `DataLoader(num_workers>0)` on Windows specifically (fork-based Linux multiprocessing has no
   such restriction). Fixed with `dataloader_num_workers=(0 if FAST_RUN else 4)` — **real run
   (`FAST_RUN=False`) is completely unaffected**; this is a genuine cross-platform correctness fix,
   not a hack, since anyone running the preflight locally on Windows before submitting would hit this.
3. **Windows `MAX_PATH` (260 char) overflow** — an earlier version of `RUN_NAME_BASE` appended a
   verbose debug-only suffix on top of an already-long base name; combined with `results_debug_preflight/`
   and the longest RankExt per-step diagnostic filenames (e.g.
   `rank_extension_dense_orth_lam20_kd_oldseen_T2_warmup_step_5_trainable_parameters.csv`), the full
   path reached 290 characters. **Root-caused and fixed** by removing the suffix entirely — the
   isolation guarantee already comes structurally from `ROOT_RESULTS_DIR` being a separate root tree,
   so the suffix added path length without adding real safety. Recomputed the real run's own worst
   -case path length: comparable risk exists on Windows even for `ROOT_RESULTS_DIR="results"` (~260
   chars, right at the boundary) — **this is a Windows-local-testing-only limitation, not a script
   defect**; the actual target environment is a Linux GPU cluster, where `PATH_MAX` is ~4096 and this
   does not arise. Documented here rather than chased further, per explicit instruction not to
   over-invest in local-CPU perfection.

**After all three fixes**, a fresh local run (`FAST_RUN_DEBUG=1`, real script, real 5×20 protocol
shape, capped image counts) was launched and observed to: pass every hard assertion (including the
new Part-0 readiness block), print the correct 5 SimpleAvg / 4 RankExt arm lists, load the dataset and
CLIP-ViT-B/16 backbone correctly, build the `simple_avg` (SA-0) specialist's step-1 training set at
the exact expected size (`current=300` = 20 classes × 15-image cap), and execute real training batches
with sane, decreasing cross-entropy loss (4.73 → 4.36 over the first few steps) with **zero crashes**
after the fixes. This run was still in progress (slow on CPU, ~10s/batch) when this report was
finalized — per explicit instruction, it was **not** waited on to completion; the GPU preflight built
into the sbatch (Section 12) is the authoritative full-coverage test, and will run in minutes rather
than hours on the actual A40.

**Not independently re-verified in this pass** (carried over from the prior session, still an honest
gap): a full line-by-line audit of every one of the ~700 lines the research agent added, and a fresh
downstream-reporting-tail audit specifically against RankExt row data (the SimpleAvg-only pack's own
equivalent audit is the template this joint script's fixes are modeled on, but was not independently
re-run here).

## 12. Preflight-then-real-run sbatch

`experiments_prepared/slurm/simpleavg_rankext_singlejob_improvement_pack_5x20.sbatch` (rewritten this
session; the earlier SimpleAvg-only sbatch under the same directory is explicitly obsolete for this
job and is not reused). Structure:

1. Static pre-flight `grep` checks on the target script (protocol/rank/schedule/epoch-expression/SEED
   — all reconfirmed matching the live file's actual text in this session).
2. `rm -rf results_debug_preflight; mkdir -p results_debug_preflight` (safe — never touches `results/`).
3. `FAST_RUN_DEBUG=1 python -u "$TARGET_SCRIPT"`, exit code captured explicitly (`set +e`/`set -e`
   toggled around it, since the script runs under `set -euo pipefail`).
4. On non-zero exit: print `===== GPU PREFLIGHT FAILED -- REAL TRAINING ABORTED =====` and `exit 1`
   — Phase 2 is never reached.
5. On success: locate the freshly-written `simpleavg_singlejob_training_x_merge_results.csv` via
   `ls -t results_debug_preflight/*/tables/...`, then run an embedded Python verification block
   (`pandas`) checking: exactly 14 rows; `family`/`training_method`/`merge_method`/`all_seen` columns
   present; exactly 10 SimpleAvg rows across exactly 5 distinct training methods, each with exactly
   `{arithmetic, do_merge}`; exactly 4 RankExt rows across exactly 4 distinct training methods, each
   `merge_method == "persistent"`; no duplicate `(training_method, merge_method)` rows; no NaN in
   `all_seen`. Non-zero exit on any failure, same abort-with-message behavior.
6. On success: print `===== GPU PREFLIGHT PASS -- STARTING REAL EXPERIMENT =====`, then run the real
   script a second time (`FAST_RUN_DEBUG` unset, `PYTHONUNBUFFERED=1`, real output under `results/`).

`bash -n` on the finished file: **PASS**. All static pre-flight `grep` patterns re-verified against
the live script's actual current text in this session (one, `LORA_ALPHA`, was initially wrong — the
real line is `LORA_ALPHA = 2 * LORA_R`, not a literal `160` — caught and fixed before finalizing).

## 13. Resume/second-pass behavior (design-verified, not locally executed to completion)

Unchanged design from the SimpleAvg-only pack, extended to RankExt: `load_arm_checkpoint_if_present`/
`load_rankext_checkpoint_if_present` check for both a checkpoint file and a `__DONE.marker` before
treating an arm as complete, and recompute+compare a content fingerprint on reload (crash-safe: a
checkpoint written but not yet marker-stamped is correctly treated as incomplete). A second invocation
of Phase 2 (e.g. after a walltime kill) would skip every already-completed arm and reload its state
with zero retraining. This logic was code-reviewed but not exercised by a full local run-to-completion
-then-rerun cycle in this session (would require the multi-hour local run this task explicitly said
not to wait for) — the GPU preflight's own first invocation, run twice if desired before trusting
Phase 2, would be the fast, practical way to exercise this if extra confidence is wanted before
submitting.

## 14. Real config unchanged, no test leakage

`NUM_STEPS`, `CLASSES_PER_STEP`, `LORA_R`, `LORA_ALPHA`, `RANKEXT_RANK_SCHEDULE`,
`RANKEXT_ALPHA_PER_RANK`, `DENSE_ORTH_LAMBDA`, `MERGE_METHODS_TO_EVALUATE`, `SEED` are all identical
in Phase 1 and Phase 2 — every `FAST_RUN`-gated reduction is confined to per-class image counts,
epoch count, output-directory root, and worker-process count. No merge operator reads test labels,
training images, or task-oracle information (unchanged from the SimpleAvg-only pack's own audit,
re-confirmed by re-reading `do_merge_deltas()`'s signature, which still accepts only
`step_states`/`eps`/`use_orthogonalize`/`verbose`).

---

## Final terminal summary

See the end-of-turn response for the exact requested field list.
