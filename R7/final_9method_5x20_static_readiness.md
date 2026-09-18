# Final 9-Method CIFAR100 5×20 — Static Readiness

**No training, no evaluation, no smoke test, and no SLURM submission were
performed to produce this report.** Every check below is either `py_compile`,
direct source inspection, or a dataset-independent execution of the
production script's own top-level configuration/assertion code (verified to
run before any dataset load or model download occurs — see Section 2).

---

## 1. Files Created

| File | Purpose |
|---|---|
| `experiments_prepared/final_9method_5x20_performance_recovery.py` | New production script (9 methods, both families 9 epochs) |
| `experiments_prepared/slurm/final_9method_5x20_performance_recovery.sbatch` | SLURM launcher (created only, not submitted) |
| `R7/final_5x20_performance_recovery_plan.md` | Design report |
| `R7/final_9method_5x20_static_readiness.md` | This report |

No historical script (`experiments_prepared/final_8arm_family_specific_5x20.py`,
`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`,
`R7/vit_lora_cifar100_full5step_n5.py`) was modified. No git operation was
performed.

---

## 2. py_compile

```
python -m py_compile experiments_prepared/final_9method_5x20_performance_recovery.py
```

**PASS** (re-run after every edit; final pass confirmed clean).

`bash -n experiments_prepared/slurm/final_9method_5x20_performance_recovery.sbatch`
— **PASS**.

---

## 3. Protocol Lock

Verified by a **dataset-independent, actual execution** of the script's own
top-level code, not just visual inspection: the file was parsed with `ast`,
and every top-level statement up to (but not including) the `dataset =
load_dataset("cifar100")` line was extracted and executed in an isolated
namespace (this boundary was chosen specifically because every protocol/
method-registry assertion in the script runs before that line — the dataset
load and CLIP checkpoint download, the only genuinely slow/network-dependent
steps, were never triggered). Result:

```
protocol              = 5x20 (100 total classes)
seed                  = 42
NUM_STEPS == 5: PASS
CLASSES_PER_STEP == 20: PASS
NUM_CLASSES == 100: PASS
class_splits == [[0..19],[20..39],[40..59],[60..79],[80..99]]: PASS
active_rankext_rank_schedule() == [16, 32, 48, 64, 80]: PASS
USE_RANKEXT_RANK_SCHEDULE_WIDE: False (confirmed, and asserted False twice)
```

All hard assertions specified in the task brief's Section 1 are present in
the script (not merely implied) and were confirmed to evaluate `True` by
this execution, not by reading the assertion text alone.

---

## 4. Exact 9-Method Set

Same execution as Section 3, extended through the method-registry assertion
block:

```
ACTIVE_METHOD_NAMES (9): [
  'rank_extension',
  'rank_extension_factor_orth_lam50',
  'rank_extension_factor_orth_lam50_fullkd_T2_protect30',
  'rank_extension_fullkd_T2_protect30',
  'simple_avg',
  'simple_avg_dense_orth_lam20',
  'simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup',
  'simple_avg_kd_oldseen_T2_warmup',
  'simple_avg_kd_oldseen_T4_warmup',
]
set(active_methods) == EXPECTED_METHODS: PASS
len(active_methods) == 9: PASS
SimpleAvg count == 5: PASS
RankExt count == 4: PASS
```

This exactly matches the task brief's Section 6 `EXPECTED_METHODS` set,
verified by set equality at module-configuration time, not by manual
comparison.

**Two real bugs were found and fixed during this verification** (not merely
theoretical risks — both raised `AssertionError` on the first execution
attempt, before being fixed):
1. A stale `assert len(ACTIVE_METHOD_NAMES) == 8` (and a sibling
   `assert _expected_final_configs == 8`) inherited from the 8-arm script,
   corrected to 9.
2. A KD-temperature-sweep consistency check (`assert sorted(temps) ==
   sorted(KD_TEMPERATURES)`, originally written to catch a base_method
   silently missing a temperature from a shared sweep list) incorrectly
   flagged the new, deliberately-standalone T=4 SimpleAvg arm as a mismatch
   against the global `KD_TEMPERATURES=[2.0]` list. Fixed by excluding that
   arm's `base_method` from the shared-sweep check and asserting its
   temperature is exactly 4.0 directly instead.

Both fixes were verified by re-running the same dataset-independent
execution to a clean pass afterward.

---

## 5. Epoch Verification

```
LORA_EPOCHS == 9: PASS
RANKEXT_EPOCHS == 9: PASS
```

Asserted three times in the script (immediately after each constant's own
definition, once more in the post-method-registration block, and once more
in the pre-existing readiness-verification block that was updated from its
stale `== 6` check) — all three confirmed to hold under actual execution.
No `FAST_RUN` conditional exists on these two constants; they are flat,
unconditional values.

---

## 6. SimpleAvg Configuration

Confirmed by the same execution:

```
simple_avg_rank       = 80
simple_avg_alpha      = 160
simple_avg_scaling    = 2.0
simple_avg_targets    = ['q_proj', 'v_proj']
SIMPLEAVG MERGES: ['arithmetic']   (DO-Merge: absent from MERGE_METHODS_TO_EVALUATE, asserted)
replay: assert all(not uses_replay ...) -- PASS
```

---

## 7. SimpleAvg T2/T4 Verification

```
simple_avg_kd_oldseen_T2_warmup:  uses_kd=True, kd_class_scope='old_seen', kd_temperature=2.0, kd_weight=1.0, kd_warmup_epochs=1.0
simple_avg_kd_oldseen_T4_warmup:  uses_kd=True, kd_class_scope='old_seen', kd_temperature=4.0, kd_weight=1.0, kd_warmup_epochs=1.0
```

Every field identical between the two except `kd_temperature` — confirmed by
direct field-by-field assertion, not visual diff. T² scaling is applied via
`self.kd_temperature ** 2` inside the shared, generic trainer `compute_loss`
code (`masked_kd_loss()` / `IndependentLoraOrthTrainer`) — confirmed by
source inspection; no per-temperature special case exists anywhere in the
KD loss path.

---

## 8. DenseOrth20 Verification

```
simple_avg_dense_orth_lam20:                          uses_dense_orth=True, lambda_orth=20.0
simple_avg_dense_orth_lam20_kd_oldseen_T2_warmup:      uses_dense_orth=True, lambda_orth=20.0, uses_kd=True (T=2, old_seen, weight=1.0, warmup=1ep)
No RankExt arm uses DenseOrth: PASS (assert not any(... family=='rank_extension' ... uses_dense_orth))
```

DenseOrth's own loss formula (`compute_rankext_dense_orth_components`/the
SimpleAvg dense-delta cosine² path) is unchanged from job 4970580 — not
re-implemented for this script.

---

## 9. RankExt Full-KD Verification

```
rank_extension_fullkd_T2_protect30:                          kd_class_scope='full', kd_temperature=2.0, kd_weight=1.0, kd_warmup_epochs=0.0
rank_extension_factor_orth_lam50_fullkd_T2_protect30:        kd_class_scope='full', kd_temperature=2.0, kd_weight=1.0, kd_warmup_epochs=0.0
```

`kd_class_scope == "full"` (not `"old_seen"`) confirmed for both by
assertion — this codebase's own scope string for full-100-way KD is `"full"`
(not the task brief's illustrative `"full_100"` label; the assertion checks
the actual, unambiguous value the running trainer code branches on, per
`masked_kd_loss()`/`DeltaOrthRankExtensionTrainer.compute_loss()`'s own
`if self.kd_class_scope == "old_seen": ... else: <full 100-way KL> ...`
dispatch — confirmed by direct source read, not inferred). No KD-weight
warmup confirmed (`kd_warmup_epochs == 0.0` for both, matching Exp1's
flat-weight convention exactly — the mechanism didn't exist historically, so
"no warmup" is the correct reproduction, not a simplification).

---

## 10. FactorOrth50 Verification

```
rank_extension_factor_orth_lam50:                       uses_factor_orth=True, lambda_orth=50.0, uses_dense_orth=False
rank_extension_factor_orth_lam50_fullkd_T2_protect30:   uses_factor_orth=True, lambda_orth=50.0, uses_dense_orth=False
No SimpleAvg arm uses FactorOrth: PASS
rank_extension_factor_orth_lam50 (non-KD) NOT in RANKEXT_PROJECTED_PROTECT_METHODS: PASS
rank_extension_factor_orth_lam50 (non-KD) NOT in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS: PASS
```

FactorOrth's own geometry (A/B factor cosine-overlap against accumulated
frozen blocks, 1-epoch λ warmup) is unchanged from job 4970580's already-
verified implementation — confirmed identical to
`R7/vit_lora_cifar100_full5step_n5.py`'s own version during the design
audit referenced in the plan document (Section 6/12).

---

## 11. Protect30 Verification

```
rank_extension_fullkd_T2_protect30 in RANKEXT_PROJECTED_PROTECT_METHODS: PASS
rank_extension_factor_orth_lam50_fullkd_T2_protect30 in RANKEXT_PROJECTED_PROTECT_METHODS: PASS
RANKEXT_PROJECTED_FEATURE_PROTECT_WEIGHT == 30.0: PASS (unchanged constant)
```

The mechanism itself (`compute_old_semantic_subspace`, SVD-based old-class
discriminative subspace, projected-drift penalty) is the pre-existing,
unmodified function already present (but previously always inactive) in
`final_8arm_family_specific_5x20.py` — confirmed byte-identical to
`R7/vit_lora_cifar100_full5step_n5.py`'s own version by direct comparison
during preparation of this script (both define the identical `W_hat`/
`W_tilde`/SVD/rank-tolerance/`P_old` construction, and the identical
projected-drift loss formula in `compute_loss`). No new implementation was
written; only the two new method names were added to the activation set.

---

## 12. New-Block Warmup Verification

```
rank_extension_fullkd_T2_protect30 in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS: PASS
rank_extension_factor_orth_lam50_fullkd_T2_protect30 in RANKEXT_NEW_BLOCK_WARMUP_DISABLED_METHODS: PASS
rank_extension (plain) NOT in that set: PASS (new-block warmup stays enabled)
rank_extension_factor_orth_lam50 (non-KD) NOT in that set: PASS (new-block warmup stays enabled)
```

Verified per-method, not imposed uniformly — matching the task brief's
explicit instruction to check source behavior for non-KD arms rather than
assume one setting for all of RankExt.

---

## 13. Classifier Restoration

`restore_protected_classifier_rows()` (inherited unmodified from
`final_8arm_family_specific_5x20.py`) uses `tensor[idx] = value`
(`__setitem__`/`index_put_`, genuinely in-place) for both `weight` and
`bias` — confirmed by direct source read; the historical
`tensor[idx].copy_(value)` advanced-indexing no-op is **absent** from this
file. `classifier_restoration_mode` is additionally recorded in the extended
checkpoint fingerprint (Section 16) as the literal string
`"fixed_inplace_assignment"`, so any future checkpoint reload is tagged with
which restoration semantics produced it. Diagnostics
(`protected_weight_max_abs_diff`, `protected_bias_max_abs_diff`) are
computed and asserted `< 1e-6` in the reporting section, unchanged from job
4970580's already-verified mechanism.

---

## 14. Logging

Per-batch, per-(method, step, epoch) logging is unchanged from job 4970580's
already-complete schema (`ce_loss`, `total_loss`, `kd_*`, `orth_*`,
`protect_*` fields are all present in the shared trainer row-dict — the
`protect_loss`/`weighted_protect`/`protect_weight` fields were already being
computed and logged by the inherited trainer code, simply never populated
with non-zero values before this script activated `protect_weight` for two
methods). The per-`(method, step, epoch)` aggregation table
(`final_9method_loss_by_method_step_epoch.csv`) was extended to additionally
average `protect_loss`/`weighted_protect`/`protect_weight` when present.
RankExt CL retention/plasticity trajectory and classifier-restoration
diagnostics tables are unchanged mechanisms, renamed to
`final_9method_*` output paths.

---

## 15. Reporting Units

The prior units bug (all_seen`/`first_step`/`later_steps_mean`/`old_new_gap`
stored as 0–1 fractions while `restricted_mean` was already 0–100) is fixed
at the summary-table-construction site: `all_seen`, `first_step`, and
`later_steps_mean` are now explicitly multiplied by 100.0 before being
placed in the summary row, and `old_new_gap` is recomputed from the
already-converted percentage values (not the raw fractions). A runtime
sanity assertion runs **before** the table is written to disk: every value
in `{all_seen, restricted_mean, first_step, later_steps_mean}` must be
non-negative and within `[-100, 100]` (the same bound also applies to the
signed `old_new_gap`), or the script raises `AssertionError` naming the
offending column and values rather than silently writing a bad table.

---

## 16. Checkpoint Safety

`RUN_NAME_BASE = "cifar100_5x20_final_9method_performance_recovery_seed{SEED}"`
— confirmed distinct from every prior run's checkpoint root
(`..._final_8arm_family_specific_seed42...`,
`..._simpleavg_rankext_singlejob_improvement_pack_seed42...`). This was
confirmed as a plain string literal via the same dataset-independent
execution (`run_name_base = 'cifar100_5x20_final_9method_performance_recovery_seed42'`).

`compute_method_config_fingerprint()` was extended (task brief Section 30)
to additionally cover: `lora_alpha`, `protect_weight`,
`rankext_new_block_warmup_enabled`, `merge_method`, and
`classifier_restoration_mode`, on top of the pre-existing fields (family,
rank, epochs, KD scope/weight/temperature/warmup, orth mode+lambda, target
modules, seed, protocol shape, rank schedule). A checkpoint whose stored
fingerprint does not match the current config is rejected and the arm is
retrained from scratch — unchanged resume-safety mechanism, now covering
every field this script's new arms actually introduce.

---

## 17. SLURM Launcher

| Field | Value | Matches task brief? |
|---|---|---|
| `cd` target | `/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project` | Yes |
| Python | `/nfsd/lttm4/tesisti/shahrampour/conda_envs/my_env_clean/bin/python` | Yes |
| Script | `experiments_prepared/final_9method_5x20_performance_recovery.py` | Yes |
| `--partition` | `allgroups` | Yes |
| `--gres` | `gpu:a40:1` | Yes |
| `--cpus-per-task` | `4` | Yes |
| `--mem` | `32G` | Yes |
| `--time` | `18:00:00` | Yes |
| `--output` | `final_9method_5x20_performance_recovery_%j.out` | Yes |
| `--error` | `final_9method_5x20_performance_recovery_%j.err` | Yes |
| `unset FAST_RUN_DEBUG` / `unset REPLICATION_SEED` | present | Yes |

`bash -n` syntax check: **PASS**. Not submitted (`sbatch` was never invoked).

---

## 18. Remaining Runtime Risks

These cannot be resolved by static validation and are carried forward as
known, disclosed risks (not blockers per Section 37's rule):

1. **Arms 7/9's exact configuration (full KD + protect30 + no new-block
   warmup, 9 epochs, corrected restoration) has never been executed before**
   — only its individual mechanisms have been separately verified safe
   (Section 9–14 above confirm the *code path* is correct and identical to
   already-proven-safe historical code; they cannot confirm the *trained
   outcome*).
2. **18-hour walltime is an estimate**, not a timed measurement of this exact
   arm/epoch mix.
3. **GPU-specific behavior (fp16, CUDA memory) was not exercised** — this
   machine has no GPU; static checks confirm `USE_FP16 = torch.cuda.is_available()`
   is unchanged from the already-GPU-verified job 4970580 pipeline, not that
   it behaves correctly on hardware this session cannot access.
4. **The two new SLURM-launcher-affecting constants (epochs, method count)
   were the only ones changed from a working, previously-submitted launcher
   template** (`final_8arm_family_specific_5x20.sbatch`) — no other resource
   field was altered beyond `--time` (12h→18h) and the job name/script/output
   paths.

None of these are treated as blockers per the task's explicit instruction
not to withhold readiness for the absence of a smoke test or live execution.

---

## 19. Final Verdict

**READY TO RUN: YES**

Every static/configuration check specified in the task brief was performed
and passed, including two real bugs discovered and fixed during preparation
(Section 4). No unresolved code or configuration issue remains.
