# R8 Final Static Readiness Audit

**Status: STATICALLY AUDITED AND SELF-TESTED; full execution remains to be validated by the
actual R8 run.** No CIFAR loading, CLIP loading, GPU training, smoke training, or cluster
submission was performed for this audit — every finding below comes from source tracing,
invariants/assertions, and synthetic self-tests (`--mode selftest`, plus new checks added by this
pass). The previously-running bounded smoke test was cancelled per instruction and its outcome is
not used as evidence anywhere in this report.

Target file: `experiments_prepared/supervisor_regularizer_repair_both_families_r8.py`
Historical reference: `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` (R7 —
**not modified**, verified via `git status`). Thesis Chapters 1–3 also not modified.

This audit found and fixed **five implementation bugs** that would have made a real cluster
submission either crash immediately, silently run on the wrong hardware, or complete without
producing any recoverable output. It also found two items that were not bugs — a behavioral
confound and a metrics gap — and initially left both unresolved, per instructions, pending an
explicit decision.

**UPDATE (this pass): both remaining items have now been explicitly decided and implemented/
documented** — see the "Decision 1" and "Decision 2" amendments in Sections 3 and 12, and the
updated R7-vs-R8 comparison table in Section 16. No CIFAR/CLIP/GPU/training/smoke-test execution
was performed for this pass either — only source edits, `py_compile`, `--mode selftest`, and new
synthetic self-tests with hand-computed exact reference values.

---

## 1. `--mode train` call graph

```
CLI (--mode train)
  -> checks SELECTED_SHARED_LAMBDA is not None (else refuses)              [main(), Section 10]
  -> run_full_r8_experiment(epochs_per_step=9, batch_size=16, ...)
       -> load_cifar100() / build_transforms()
       -> build_classwise_train_val_splits(...)                             [ONCE, shared by all 8 methods]
       -> create_r8_run_directory()                                          [NEW -- Section 14 fix]
       -> for each of 8 methods:
            simple_avg family:
              run_full_method_simple_avg(...)
                -> for step_idx in 0..4:
                     train_one_step_simple_avg(...)
                       -> add_lora_simple_avg(fresh_pretrained_model())
                       -> build_simple_avg_teacher_model(step_states)  [if uses_kd and step_idx>0]
                       -> SimpleAvgCorrectedTrainer(...).compute_loss()  [per batch, per epoch]
                       -> BestEpochTracker.on_epoch_end() / .finalize()  [save/restore best state]
                       -> extract_lora_state(model)
                     step_states.append(...); previous_dense_deltas_by_module[...].append(...)
                -> simple_average_deltas(step_states) -> apply_deltas_to_base(...)   [final model]
                -> evaluate_open_restricted(...)                                     [pre-cal]
                -> calibrate_classifier_row_norms_confidence_weighted(...)
                -> evaluate_open_restricted(...)                                     [post-cal]
                -> log_accuracy_diagnostic(...) / log_final_accuracy(...)
            rank_extension family:
              run_full_method_rank_extension(...)  [analogous, persistent model + grow/freeze]
            write_r8_method_summary_csv(output_dir, results)                [NEW -- after EVERY method]
       -> verify_r8_run_complete(results, methods)                          [NEW]
       -> write_r8_diagnostic_tables(output_dir)                            [NEW]
       -> write_r8_run_config_json(output_dir, ...)                          [NEW]
  -> returns results (list of 8 per-method summary dicts)
```

Per-transition findings (bugs found are marked **BUG**, fixed in this pass):

| Transition | Finding |
|---|---|
| CLI → `run_full_r8_experiment` | Arguments correct; `SELECTED_SHARED_LAMBDA` check present and now passes (=20.0) |
| `run_full_r8_experiment` → per-method dispatch | Dispatch keys off `method_cfg["family"]`, read from the registry — no name-string branching, no stale assumption |
| model construction | **BUG (fixed): no device placement anywhere** — every model stayed on CPU regardless of cluster GPU allocation. Fixed via a new `DEVICE` constant + `.to(DEVICE)` in `fresh_pretrained_model()` (single choke point) |
| RankExt step 1 | **BUG (fixed): `train_one_step_rank_extension` unconditionally called `module.grow(...)` even at step_idx=0**, but the model was already constructed with step 1's rank baked in — `grow()`'s own assertion (`new_rank==0`) would fire immediately, crashing every RankExt method at step 1. Fixed by gating the grow call on `step_idx > 0`. Reproduced and confirmed with a synthetic (non-CIFAR/CLIP) `nn.Linear` module before and after the fix |
| training loop → `compute_loss` | **BUG (fixed): `batch` was never moved to the model's device** before the forward pass (unlike `evaluate_val_ce`/`run_inference`, which already did this) — would raise a device-mismatch `RuntimeError` the moment the model is on CUDA. Fixed inside both `compute_loss()` methods |
| optimizer construction | **BUG (fixed): `WEIGHT_DECAY = 0.0`** with an incorrect comment claiming it matched "R7's Trainer defaults" — R7 actually sets `WEIGHT_DECAY = 0.05` explicitly. Also, R7's optimizer exempts bias parameters from decay (`build_head_lr_param_groups`); R8's `build_optimizer` didn't. Both fixed; RankExt's separate `build_optimizer_from_params` (which had no decay/no-decay split at all) removed and unified onto the same `build_optimizer()` both families now share |
| best-epoch save/restore | `BestEpochTracker` keeps `best_state_dict` (deep-copied whole `model.state_dict()`, so LoRA/block params and classifier are restored atomically) and reloads it in `finalize()` before `extract_lora_state`/`add_frozen_block_from_current` — no test-set leakage, no early-epoch-0 aliasing bug (best_epoch is always 1-indexed, never falsy) |
| final model construction | SimpleAvg: `simple_average_deltas` (arithmetic mean of dense deltas) → `apply_deltas_to_base` (classifier stitched per introducing step) — matches R7. RankExt: persistent model already holds every frozen block + the finalized new block — no separate "final construction" step needed, matches R7's incremental design |
| calibration | Reads only `method_cfg["uses_kd"]` (a boolean from the registry), never a name string — immune to the new method-ID renaming (Section 11) |
| evaluation | `evaluate_open_restricted` computes open and restricted accuracy from the **same** single forward pass's logits (masking only changes the argmax candidate set) |
| metric aggregation → output | **GAP (not a bug, flagged not fixed): originally there was NO output writing at all.** Fixed with a new, incremental (per-method), never-overwrite output-persistence layer (Section 14) |
| output directories | `create_r8_run_directory()` creates `tables/`/`configs/` with `exist_ok=False` before any write, and raises `FileExistsError` on reuse — verified by self-test |

**TRAIN CALL GRAPH COMPLETE: YES** (fully traced end to end; every arrow above corresponds to an
actual call verified in source, not inferred).

---

## 2. All 8 methods — exact static configuration

| # | internal_name | family | CE | corrected KD | DenseOrth | KD teacher | orth reference | KD warmup | orth warmup | feature anchor | projected protection | new-block warmup | calibration group | final model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `simple_avg` | simple_avg | yes | no | no | none | none | n/a | n/a | **none** | **none** | n/a | global (uses_kd=False) | dense-mean + classifier stitch |
| 2 | `simple_avg_kd_oldseen_T2` | simple_avg | yes | yes | no | dense-merged prior (frozen) | none | 1 epoch | n/a | **none** | **none** | n/a | regime-grouped (uses_kd=True) | dense-mean + classifier stitch |
| 3 | `simple_avg_dense_orth` | simple_avg | yes | no | yes | none | each individual previous step delta | n/a | 1 epoch | **none** | **none** | n/a | global (uses_kd=False) | dense-mean + classifier stitch |
| 4 | `simple_avg_dense_orth_kd_oldseen_T2` | simple_avg | yes | yes | yes | dense-merged prior (frozen) | each individual previous step delta | 1 epoch | 1 epoch | **none** | **none** | n/a | regime-grouped (uses_kd=True) | dense-mean + classifier stitch |
| 5 | `rank_extension` | rank_extension | yes | no | no | none | none | n/a | n/a | **none** | **none** | ON (uniform, see §3) | global (uses_kd=False) | persistent frozen+new blocks |
| 6 | `rank_extension_kd_oldseen_T2` | rank_extension | yes | yes | no | deepcopy of previous-step persistent model (frozen) | none | 1 epoch | n/a | **none** | **none** | ON (uniform, see §3) | regime-grouped (uses_kd=True) | persistent frozen+new blocks |
| 7 | `rank_extension_dense_orth` | rank_extension | yes | no | yes | none | each individual previous frozen block | n/a | 1 epoch | **none** | **none** | ON (uniform, see §3) | global (uses_kd=False) | persistent frozen+new blocks |
| 8 | `rank_extension_dense_orth_kd_oldseen_T2` | rank_extension | yes | yes | yes | deepcopy of previous-step persistent model (frozen) | each individual previous frozen block | 1 epoch | 1 epoch | **none** | **none** | ON (uniform, see §3) | regime-grouped (uses_kd=True) | persistent frozen+new blocks |

**No hidden ninth behavior/path exists.** `R8_METHODS` is asserted (module-level `assert`, re-checked
by the self-test) to contain exactly these 8 `internal_name`s, exactly partitioned by family, and
`RankExtCorrectedTrainer`/`SimpleAvgCorrectedTrainer.compute_loss()` branch on only `uses_kd`/
`uses_dense_orth` — there is no third loss term, no third branch, and no code path reachable from
`run_full_r8_experiment` other than what is tabulated above.

**ALL 8 METHODS VERIFIED: YES**

---

## 3. Critical RankExt auxiliary-confound audit

Searched the entire file for R7's three auxiliary RankExt mechanisms:

| R7 mechanism | Present in R8? |
|---|---|
| Pretrained-backbone feature anchor (`pretrained_anchor_weight`, `compute_old_semantic_subspace`-style loss) | **Absent.** No function or parameter of this kind is defined anywhere in this file (grep for `pretrained_anchor`, `feature_anchor`, `semantic_subspace` returns zero matches outside this report) |
| Projected old-class feature protection (`protect_weight`) | **Absent.** Same search, zero matches |
| Method-dependent new-block **output** warmup (ramps how much the new rank block's forward contribution counts, during its first epoch) | **Present, but UNIFORMLY applied to all 4 RankExt methods** — see below |

### RankExt auxiliary table (all 4 methods)

| method | CE | oldseen KD | DenseOrth | feature anchor | projected feature protection | new-block output warmup | any other auxiliary loss |
|---|---|---|---|---|---|---|---|
| `rank_extension` | yes | no | no | none | none | **ON** (1 epoch) | none |
| `rank_extension_kd_oldseen_T2` | yes | yes | no | none | none | **ON** (1 epoch) | none |
| `rank_extension_dense_orth` | yes | no | yes | none | none | **ON** (1 epoch) | none |
| `rank_extension_dense_orth_kd_oldseen_T2` | yes | yes | yes | none | none | **ON** (1 epoch) | none |

### HIDDEN AUXILIARY CONFOUND: NONE (RESOLVED — Decision 1)

R7's own code (`method_rankext_new_block_warmup_epochs()`,
`experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py`) disables the new-block
output warmup **specifically and only** for the two KD RankExt methods
(`rank_extension_kd_only_T2`, `rank_extension_orth_factor_lam_50_kd_T2`), keeping it enabled for
the two non-KD methods — confirmed directly in R7's source: `"non-KD rank_extension keeps the
existing warmup, KD rank_extension disables it."` R8's `train_one_step_rank_extension` applies it
**unconditionally** (`linear_warmup_multiplier(float(epoch), 1.0, True)` — the literal `True`, not
gated on `method_cfg["uses_kd"]`) to all 4 RankExt methods, including both KD variants. This was
flagged, in an earlier version of this report, as an unresolved confound requiring a decision.

**DECISION 1 (final policy, made explicitly, not by this audit unilaterally): uniform ON for all
four RankExt methods.** Rationale: the new-block output warmup is an architecture-level
stabilization mechanism for the newly appended RankExt block, not part of KD itself. In historical
R7 it was disabled for the KD variants, but those variants also carried other KD-associated
auxiliary mechanisms (feature anchor / projected protection) that are entirely absent from R8
(Section 3's own audit, re-confirmed below). Retaining a KD-conditional warmup policy in R8 —
where those other KD-specific auxiliaries no longer exist — would itself manufacture a new,
unmotivated within-R8 confound (KD RankExt methods would differ from non-KD RankExt methods in a
mechanism unrelated to either KD or DenseOrth). Uniform ON removes that risk and keeps the within-
R8 ablation (control vs. KD vs. DenseOrth vs. both) clean along exactly the two axes R8 is actually
testing.

**Important reporting nuance, made explicit per instructions:** this means R8's KD RankExt
variants (`rank_extension_kd_oldseen_T2`, `rank_extension_dense_orth_kd_oldseen_T2`) differ from
their nominal R7-KD counterparts (`rank_extension_kd_only_T2`,
`rank_extension_orth_factor_lam_50_kd_T2`) in **two** respects, not one: the corrected KD
mask/warmup formula (the intended, documented experimental variable) **and** the new-block
output-warmup policy (an incidental, now-intentional side effect of removing R7's KD-conditional
disable). **A historical R7-KD → R8-KD result comparison must therefore never be presented as
isolating the KD formula change alone** — it also carries this warmup-policy difference. Within-R8
comparisons (R8 control vs. R8 KD vs. R8 DenseOrth vs. R8 both, all four sharing the identical
uniform warmup policy) remain clean, single-variable-at-a-time comparisons.

Verified (self-test): the single `linear_warmup_multiplier(float(epoch), 1.0, True)` call site in
`train_one_step_rank_extension` is never gated on `method_cfg["uses_kd"]` or any other per-method
flag, and appears exactly once in the function; a direct simulation confirms all 4 RankExt methods
receive the identical 0.0/1.0/1.0 (epoch 0/1/2) multiplier sequence regardless of `uses_kd`.
Feature anchor and projected feature protection are re-confirmed absent (Section 3's original
table, unchanged).

Neither option was chosen by this audit. `train_one_step_rank_extension`'s warmup line is
unchanged from before this pass, specifically so this decision stays live rather than being
silently made.

**Plain SimpleAvg hidden-auxiliary check:** confirmed — `simple_avg` and every `simple_avg_*` R8
method go through `SimpleAvgCorrectedTrainer.compute_loss()`, which has exactly two conditional
branches (`uses_kd`, `uses_dense_orth`) and no third term; there is no SimpleAvg analogue of a
feature-anchor or new-block-warmup mechanism in R7 to begin with, so there is nothing corresponding
to check for a hidden confound there. **Confirmed clean.**

---

## 4. RankExt teacher-ordering audit

**TEACHER-BEFORE-GROW: PASS** (after the fix in Section 1's table). Traced and self-tested with a
synthetic (non-CIFAR) module, for every KD step t>1:

1. Student currently represents the completed previous step — confirmed: `model` returned by
   `train_one_step_rank_extension` at the end of step t−1 has already had `add_frozen_block_from_current()`
   called on it (the function's last action), so `new_rank=0` there.
2. `teacher = copy.deepcopy(previous_model_snapshot)` where `previous_model_snapshot =
   copy.deepcopy(model)`, captured in the **orchestration loop** (`run_full_method_rank_extension`)
   *before* that step's `train_one_step_rank_extension()` call — verified this ordering is
   textually correct (the snapshot line precedes the call in source) and behaviorally correct (the
   self-test's rank table below matches the required values exactly).
3. Teacher is frozen and `.eval()` — `build_rankext_teacher_model()`: `teacher.eval()`; `for p in
   teacher.parameters(): p.requires_grad = False`; asserted again in
   `RankExtCorrectedTrainer.__init__`.
4. Student grows with the new block **only after** the teacher snapshot is taken — confirmed by
   the fixed ordering (Section 1); previously this was violated in the opposite way for step 1
   (double-grow), now fixed for all steps.
5. Growing the student cannot mutate the teacher — `copy.deepcopy` produces an independent object
   graph; confirmed by a dedicated self-test (below).
6. Teacher rank remains the previous cumulative rank — confirmed by the rank table below.

**TEACHER RANKS** (self-tested against a synthetic module, exact match):

| step | teacher rank | student rank after grow |
|---|---|---|
| 2 | 16 | 32 |
| 3 | 32 | 48 |
| 4 | 48 | 64 |
| 5 | 64 | 80 |

**TEACHER IMMUTABILITY: PASS** — dedicated self-test: after taking a snapshot, the student is
grown, frozen, and grown again (ending up with strictly more accumulated rank than the snapshot);
the snapshot's own cumulative rank is verified unchanged.

Self-test coverage added: `RankExt grow/freeze sequence matches the required teacher/student rank
table for all 5 steps (no step-1 double-grow crash)`, `teacher snapshot is immutable when the
student keeps growing after it (deepcopy independence)`.

---

## 5. Corrected KD audit

**KD MASK: PASS.**

```
C_old = union of classes_for_step(j), j < current step   [old_seen_class_ids(), derived from
                                                            classes_for_step(), never an
                                                            independent range formula]
teacher_old = teacher_logits[:, C_old]   student_old = student_logits[:, C_old]   (index_select,
                                                            BEFORE softmax)
KD = KL(softmax(teacher_old/T), log_softmax(student_old/T)) * T^2
```
implemented once, in `masked_kd_loss()`, called identically from both trainers.

- Current classes excluded before softmax: yes — `idx` is built exclusively from `old_class_ids`,
  which by `old_seen_class_ids()`'s own construction never includes `current_step_class_ids()`.
  `assert_kd_mask_excludes_current_and_future()` checks this for all 5 steps (self-test: PASS).
- Future classes excluded before softmax: same mechanism, same assertion.
- Step 1 KD inactive: `old_seen_class_ids(0) == []` → `masked_kd_loss` returns a zero tensor
  without touching student/teacher logits at all when `old_class_ids` is empty.
- Steps 2–5 active for KD variants: `if self.uses_kd and self.teacher_model is not None and
  len(self.old_class_ids) > 0`.
- No replay: training data per step comes only from that step's own `classes_for_step(step_idx)`
  filter — no prior-step data ever enters a training loader.
- **No historical full-100-way KD enters corrected variants**: self-test confirms `F.kl_div`
  appears in exactly one production call site (inside `masked_kd_loss`) — neither trainer nor
  either orchestration function calls it directly.
- Same KD function used for SimpleAvg and RankExt: literally the same `masked_kd_loss()` Python
  function, called from both `compute_loss()` methods (verified by source inspection in the
  self-test).
- Same one-epoch KD-weight warmup used for both: `KD_WARMUP_EPOCHS = 1.0`, `KD_WARMUP_ENABLED =
  True`, one module-level definition, read by both trainers via `linear_warmup_multiplier(...)`.

`F.kl_div` usage: `F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")` — standard
PyTorch convention (`input`=log-probabilities, `target`=probabilities) computes
`KL(target‖input) = KL(teacher‖student)`, the documented/expected direction; `reduction=
"batchmean"` matches R7's KD convention. Then multiplied by `T**2`, matching Hinton-style KD
scaling.

**FULL-100 KD ABSENT FROM CORRECTED METHODS: YES**

---

## 6. DenseOrth audit

**DENSEORTH: PASS.**

```
delta_t = scaling * B_t @ A_t          (current step's/block's own dense update)
cos      = <delta_t, delta_i>_F / (||delta_t||_F ||delta_i||_F + eps)     (per target module m,
                                                                            per previous i < t)
L_orth   = mean over {m, i<t} of cos^2
```

Verified:
- **Whole-matrix granularity**: `dense_cosine_sq()` operates on the full `[out_features,
  in_features]` dense delta per target module — never reduced to per-column or per-row.
- **q_proj + v_proj only**: `TARGET_MODULES = ["q_proj", "v_proj"]`, the only modules
  `add_lora_simple_avg`/`add_rankext_lora` ever touch.
- **No per-column version**: confirmed by direct reading — `dense_cosine_sq` takes the full
  tensors, no `column_decouple_delta`/`column_normalize`-style per-column split exists anywhere in
  this file.
- **No A/B factor-space orthogonality, no averaged A/B reference**: confirmed in Section 2's
  audit — `average_factor_reference_state`/`compute_independent_lora_factor_orth_components`
  (R7's factor-space mechanism) are never `def`-ined in this file (self-test: PASS).
- **Each previous incremental dense delta kept individually**: SimpleAvg —
  `previous_dense_deltas_by_module[module].append(delta.clone().detach())` after every step,
  never replacing/averaging; RankExt — `self.frozen_blocks` is a list, appended to, never
  concatenated into one blob (the entire point of the new `GrowingRankLoRALinearDenseOrth` class,
  vs. R7's `GrowingRankLoRALinear`).
- **Previous deltas detached**: `dense_orth_penalty()` asserts `not d.requires_grad` for every
  entry in `previous_deltas`, raising loudly otherwise (verified by self-test: a non-detached
  tensor passed in raises `AssertionError`).
- **Current delta retains gradient**: computed live from `module.lora_A["default"].weight`/
  `module.lora_B["default"].weight` (SimpleAvg) or `self.A_new`/`self.B_new` (RankExt) — never
  detached; verified by a self-test backprop check.
- **Lambda = 20.0**: `SELECTED_SHARED_LAMBDA = 20.0` (module constant, self-test verifies both
  families' expected gradient ratio at this value falls inside [0.05, 0.5]).
- **One-epoch linear orth warmup**: `ORTH_WARMUP_EPOCHS = 1.0`, `ORTH_WARMUP_ENABLED = True`,
  applied identically via `linear_warmup_multiplier()` for every DenseOrth variant in both
  families (no asymmetry — unlike R7's FactorOrth, and unlike the new-block-warmup confound in
  Section 3, which is a *different* warmup mechanism).
- **Identical conceptual function in both families**: `dense_orth_penalty()`/`dense_cosine_sq()`
  are the literal same functions; SimpleAvg's trainer compares the current independent adapter's
  delta against each previous independent step's delta, RankExt's compares the current new
  block's delta against each previous frozen block's delta — same formula, family-specific only in
  which tensors are supplied.

**DENSEORTH LAMBDA: 20.0**

---

## 7. Plain control integrity

**PLAIN CONTROLS CLEAN: YES.**

`simple_avg` (`uses_kd=False, uses_dense_orth=False`) and `rank_extension` (same) are the only two
methods with both flags False. Traced both `compute_loss()` methods: `teacher_model` is only ever
constructed `if method_cfg["uses_kd"]` (checked at the `train_one_step_*` call site, before the
trainer is even constructed) — for the plain controls it is always `None`, and `self.uses_kd` is
`False`, so the KD branch's `if self.uses_kd and ...` never evaluates true regardless of any other
state. Same structure for `uses_dense_orth`. Neither plain control can receive KD or DenseOrth
through any code path.

**Baseline-specific historical mechanism that remains active:** plain `rank_extension` keeps the
new-block output warmup ON, matching R7's own plain-`rank_extension` configuration exactly (R7:
`rankext_new_block_warmup_enabled: true` for the non-KD methods) — this part of Section 3's finding
is *not* a confound for the plain control itself, only for the two RankExt KD variants. Plain
`simple_avg` carries no auxiliary mechanism of any kind, matching R7's plain SimpleAvg (which also
has none).

---

## 8. SimpleAvg end-to-end static audit

Traced `run_full_method_simple_avg` → `train_one_step_simple_avg` → `extract_lora_state` →
`simple_average_deltas` → `apply_deltas_to_base` → calibration → evaluation:

- **Final dense delta = mean of independently trained step deltas, NOT factor averaging** —
  `simple_average_deltas()`: `torch.stack(vals, dim=0).mean(dim=0)` where each `val` is a full
  `scaling*(B@A)` dense delta extracted per step; no `lora_A`/`lora_B` factor is ever averaged
  anywhere in this file (confirmed by the same "no factor-space function defined" check in
  Section 2/6).
- **Final classifier rows/biases taken from the introducing/training step exactly as in R7** —
  `apply_deltas_to_base()`: `for step_idx, state in enumerate(step_states): classes =
  classes_for_step(step_idx); ... model.classifier.weight[c].copy_(w[c])` — each class's row comes
  from the state extracted at the step that trained it, verbatim, never blended.
- **Running teacher and final evaluation merge are not accidentally conflated**:
  `build_simple_avg_teacher_model(step_states)` is called mid-loop, on the `step_states`
  accumulated *so far* (i.e., strictly the previous steps), producing a teacher used only inside
  that step's training; the *final* merge (`simple_average_deltas(step_states)` after the loop
  completes, over *all 5* steps) is a separate call, on a separate (later, complete) `step_states`
  list, assigned to a separate variable (`final_model`) never reused as a teacher. The two never
  share a variable or get passed into each other.

---

## 9. RankExt end-to-end static audit

Traced the grow/freeze/forward sequence:

- Step 1 rank 16 → freeze → step 2 grow +16 → ... → final rank 80: confirmed exactly by Section 4's
  rank table (post-fix).
- **Previous blocks frozen**: `add_frozen_block_from_current()` creates `nn.Parameter(...,
  requires_grad=False)` for both `A_frozen`/`B_frozen`; self-test confirms `requires_grad=False`
  immediately after freezing.
- **Current block trainable**: `A_new`/`B_new` created via plain `nn.Parameter(torch.zeros(...))`
  (default `requires_grad=True`); self-test confirms this both at construction and after `grow()`.
- **Current forward includes frozen history + current block**: `forward()`: `out = base_out; for
  (A,B) in self.frozen_blocks: out = out + scaling*...; if new_rank>0: out = out +
  warmup*scaling*...` — sums every frozen block plus the new block, never skips any.
- **`current_new_delta()` only refers to the new block**: `scaling*(B_new@A_new)`, nothing else.
- **`previous_block_deltas()` returns individual frozen blocks**: a list comprehension over
  `self.frozen_blocks`, one dense delta per entry, never summed/concatenated.
- **No historical block is overwritten**: `add_frozen_block_from_current()` only ever *appends* to
  `self.frozen_blocks`; there is no assignment that replaces an existing list entry anywhere in
  this file.
- **Final model uses all blocks**: `forward()`'s loop iterates the complete `self.frozen_blocks`
  list, which by construction contains one entry per completed step.

**RANKEXT GROW/FREEZE: PASS** (post-fix; pre-fix this was a guaranteed step-1 crash, Section 1).

---

## 10. Best-epoch / validation audit

- **Classwise held-out validation semantics match R7**: `build_classwise_train_val_splits()` — per
  class, shuffle with seed `SEED + class_id`, hold out `VALIDATION_PER_CLASS=25` (matches R7's own
  constant and per-class-shuffle-then-holdout mechanism exactly; see Section 13 for the
  performance-only rewrite, which preserves this exactly, self-tested for bit-for-bit equivalence).
- **Validation CE only determines best epoch**: `BestEpochTracker.on_epoch_end(epoch, val_ce,
  model)` — the sole comparison is `if val_ce < self.best_val_ce`; no other loss term (KD, orth)
  ever enters this comparison, for either family, matching R7's stated "keep best-epoch selection
  simple" policy exactly (an audit question, not a request to change it — unchanged here).
- **Best state is actually stored**: `self.best_state_dict = copy.deepcopy(model.state_dict())`,
  only on improvement.
- **Best state is actually restored before step finalization**: `finalize()`:
  `model.load_state_dict(self.best_state_dict)`, called before `extract_lora_state`/
  `add_frozen_block_from_current` in both `train_one_step_*` functions.
- **No test-set information enters selection**: `val_loader` is built exclusively from
  `val_source` (the held-out slice of the TRAIN split); `dataset["test"]` is referenced only in
  the `eval_loaders` construction, which happens strictly *after* the per-step training loop, for
  final evaluation only.
- **No regularization loss contaminates the best-epoch criterion**: `evaluate_val_ce()` computes
  `out.loss` from a plain forward pass (`F.cross_entropy` inside `CLIPVisionForCIFAR100.forward`),
  never calling `compute_loss()` (the function that adds KD/DenseOrth) — matches R7's policy,
  unchanged.
- **Classifier state restored consistently with LoRA/RankExt state**: `model.state_dict()`
  captures the whole model (LoRA/block params + classifier) as one atomic snapshot; there is no
  separate classifier-only or adapter-only restore path.

**BEST-EPOCH SAVE/RESTORE: PASS**

---

## 11. Calibration method-ID audit

`calibrate_classifier_row_norms_confidence_weighted(model, val_ce_by_step, uses_kd=
method_cfg["uses_kd"], ...)` — the **only** input that determines KD-vs-non-KD grouping behavior is
the `uses_kd` **boolean**, read directly from the same registry dict entry as `internal_name`. There
is no string-matching, no hardcoded method-name list, and no separate lookup table anywhere in the
calibration code that could fall out of sync with a renamed method ID. Concretely:

- `simple_avg_kd_oldseen_T2`, `simple_avg_dense_orth_kd_oldseen_T2`, `rank_extension_kd_oldseen_T2`,
  `rank_extension_dense_orth_kd_oldseen_T2` all have `uses_kd=True` in `R8_METHODS` → correctly
  routed to the regime-grouped (`[[step1],[steps2-5]]`) + confidence-boost path.
- The other 4 methods have `uses_kd=False` → correctly routed to the single-group (flat mean-match)
  path.

Since renaming a method's `internal_name` cannot change its `uses_kd` value (they live in the same
dict literal, set once, never derived from the name string), **a renamed ID cannot accidentally
receive the wrong calibration boost** — this is true by construction, not merely by the current
values happening to be correct.

**CALIBRATION NEW-METHOD-ID CLASSIFICATION: PASS**

---

## 12. Metric audit (DECISION 2 — implemented, this pass)

| Metric | Status |
|---|---|
| All-seen open accuracy | **Computed** — `evaluate_open_restricted()["all_seen_open"]` |
| Per-step open accuracy | **Computed** — `evaluate_open_restricted()["per_step_open"]` |
| Restricted/task-oracle accuracy | **Computed**, from the *same* logits as open accuracy, masking only the eligible-class set before argmax (`restricted_argmax_accuracy`) — confirmed single forward pass, no second model call. Not confused with specialist accuracy: `evaluate_single_step_accuracy()` (the SimpleAvg diagonal) is a *separate*, OPEN-only number on the pre-merge specialist, never masked/restricted, and never written into the `per_step_restricted` field |
| Forward transfer | **Correctly absent, and confirmed still absent** — not resurrected; self-test greps for the quoted form (as a real output key), excluding the one explanatory comment that documents its deliberate absence |
| **BWT (backward transfer)** | **Implemented, ported from R7's `compute_backward_transfer()` exactly** |
| **Average forgetting** | **Implemented for RankExt, ported from R7's `compute_average_forgetting()` exactly; NaN by construction for SimpleAvg (also R7's own convention)** |

**METRICS: PASS.** Ported directly from R7's canonical implementation — **not reinvented**:

- `compute_backward_transfer(diagonal_map, final_map)` — copied verbatim (same formula: mean over
  steps ≠ last of `final_map[s] - diagonal_map[s]`, same NaN edge-case handling) from R7's function
  of the identical name.
- `compute_average_forgetting(stepwise_task_accuracies)` — copied verbatim (same nested-dict
  traversal, same `best_acc - final_acc` per non-final task, same mean) from R7's function of the
  identical name.
- `evaluate_seen_step_accuracies(model, upto_step_idx, ...)` (RankExt) — ported from R7's function
  of the identical name: OPEN accuracy of the **current persistent, UNCALIBRATED** model on every
  task introduced so far, on the TEST split, called mid-loop immediately after each step's
  training finishes (identical call-site timing to R7).
- `evaluate_single_step_accuracy(model, step_idx, ...)` (SimpleAvg) — ported from R7's function of
  the identical name: OPEN accuracy of the pre-merge specialist on its own class group, on the
  TEST split.

**RankExt** (persistent trajectory — the "true" BWT/forgetting semantics): `stepwise_task_
accuracies[model_step][task_step]` is built incrementally, one row per step, via
`evaluate_seen_step_accuracies()` called on the **uncalibrated** model immediately after that
step's `train_one_step_rank_extension()` returns (matching R7's own call-site timing exactly —
R7's `stepwise_task_accuracies[step_idx] = seen_acc` also happens mid-loop, before calibration).
`diagonal_accuracy = {s: stepwise_task_accuracies[s].get(s, nan) for s in stepwise_task_
accuracies}` (R7's own dict-comprehension pattern, not a separate evaluation). `avg_forgetting =
compute_average_forgetting(stepwise_task_accuracies)` — uses the **uncalibrated** final-step
entry of the matrix as its "final" reference, exactly as R7 does (R7 never recomputes a calibrated
version of this specific dict either — this asymmetry vs. BWT below is R7's actual behavior,
preserved, not "fixed"). `backward_transfer = compute_backward_transfer(diagonal_accuracy,
post_cal["per_step_open"])` — the "final" side here **is** post-calibration (reusing
`evaluate_open_restricted()`'s already-computed `per_step_open`, mathematically identical to R7's
separate `evaluate_per_step_accuracy(final_rank_model, ...)` call on the calibrated model).

**SimpleAvg** (does NOT get a persistent-trajectory BWT/forgetting — a structural surrogate only,
labeled as such): `specialist_diagonal_accuracy[step_idx] = evaluate_single_step_accuracy(model,
step_idx, ...)`, called on each step's own (pre-merge) specialist right after best-epoch restore,
before it is discarded — R7's own "closest honestly-available equivalent" substitute for a true
diagonal, since SimpleAvg's steps are independent specialists with no evolving persistent model to
build a real trajectory from. `backward_transfer = compute_backward_transfer(specialist_diagonal_
accuracy, post_cal["per_step_open"])` — the **same** `compute_backward_transfer()` function
RankExt uses (not a separate formula), fed the structural-surrogate diagonal instead of a true
one. **`avg_forgetting = float("nan")`, unconditionally, by construction** — never computed from a
fabricated persistent trajectory, matching R7's own `"avg_forgetting": np.nan` for every SimpleAvg
method exactly.

**Metric storage / output**: `write_r8_method_summary_csv()` and the new
`write_r8_final_summary_json()` both carry `backward_transfer`/`avg_forgetting` for every method;
`write_r8_diagnostic_tables()` additionally writes the full RankExt stepwise trajectory matrix
(`rankext_stepwise_accuracy_by_method.csv`, one row per (method, model_step, task_step)) so the
underlying trajectory — not just the summary scalar — is recoverable. SimpleAvg's `avg_forgetting`
NaN survives pandas→CSV (empty field, R7's own convention) and Python's `json.dump` (literal `NaN`
token, `allow_nan=True` default) without being coerced to `0.0`, `null`, or a string. No test-set
result affects training or best-epoch selection anywhere in this new code (every new evaluation
call happens strictly after that step's `train_one_step_*` call returns). All new evaluations are
read-only (`@torch.no_grad()` via `run_inference()`), never touching gradients or optimizer state.

**Self-tested with exact, hand-computed synthetic reference values** (not just "runs without
crashing"): a 5-step RankExt BWT example (expected −0.1575), a 5-task RankExt forgetting example
(expected 0.2125), and a SimpleAvg BWT-surrogate example (expected −0.17) — see the report's
companion self-test output for the full worked arithmetic; all three match the ported functions'
actual output to floating-point precision.

---

## 13. Data/split performance audit

**Before this audit**: `build_classwise_train_val_splits()` called `train_ds.filter(lambda ex:
label==cls)` once **per class** — 100 separate full-table scans of the 50,000-row training set,
each invoking a Python predicate per row. An earlier (now-cancelled) bounded smoke test observed
~27s per class-filter call, i.e. ~45 minutes of pure setup overhead before a single training batch
would run — for every full `--mode train` invocation.

**Exactly how many full-dataset passes occurred**: 100 (`len(all_class_ids)`, one per class, each a
full 50,000-row scan). Splits were built **once globally**, shared across all 8 methods (not once
per method/step) — that part was already correct; the inefficiency was entirely inside the
single "once globally" call.

**Fix implemented** (authorized explicitly, bit-for-bit membership preserved): replaced the
100-scan loop with **one** full read of the label column (`train_ds[label_col]`, a single columnar
access, not a per-row Python callback) followed by cheap `numpy.nonzero`+`.select()` per class on
the already-tiny (~500-row) resulting subsets, then the identical `.shuffle(seed=seed+cls)` +
holdout logic as before.

**Equivalence proof**: (1) `.filter()` and index-based `.select()` on class-matching positions both
preserve original row order for matching examples, so the two approaches hand `.shuffle()` an
identical row sequence per class; (2) `Dataset.shuffle(seed=...)`'s permutation is a pure function
of (length, seed), not content — so the resulting permutation, and hence the train/val partition,
is identical either way. **Self-tested directly**: a synthetic (non-CIFAR) 4-class, 12-examples-
per-class dataset run through both the optimized function and an independently-reimplemented
reference (kept only for this test, never used in production) produces bit-for-bit identical TRAIN
and VAL row memberships (verified by comparing `(row_id, label)` pairs, sorted, for exact equality)
— `--mode selftest`: PASS.

No scientific configuration changed: same seed formula (`SEED + class_id`), same
`VALIDATION_PER_CLASS=25`, same shuffle-then-holdout order.

**DATA-SPLIT STARTUP PERFORMANCE: OPTIMIZED**

---

## 14. Output / resume / failure audit

**Before this audit**: `run_full_r8_experiment()` returned an in-memory list and nothing else —
zero file output. Every diagnostics accumulator in Section 8.7 of the preparation report was
populated in memory and never persisted. A real multi-hour cluster job would have completed (or
crashed) leaving **nothing** recoverable on disk.

**Fixed, this pass** (Section 8.86 of the script):
- `create_r8_run_directory()`: unique `results_r8/<run_tag>/{tables,configs}/` directory, `run_tag`
  timestamp-based by default, `exist_ok=False` at every `os.makedirs` call — **never** overwrites
  an existing run (raises `FileExistsError`), and lives under a directory name (`results_r8/`)
  that can never collide with R7's own `results/<run_name>_<timestamp>/` convention or any R7 run.
- `write_r8_method_summary_csv()`: called **after every method**, not only at the end — rewrites
  `tables/final_metrics_all_methods.csv` from the results accumulated so far. If method k+1 raises,
  methods 1..k's results are already safely on disk.
- `verify_r8_run_complete()`: asserts the final `results` list's method names exactly match the
  requested method list, in order; **raises** (does not silently pass) if any are missing —
  self-tested both for the pass and the raise case.
- `write_r8_diagnostic_tables()`: dumps every non-empty Section 8.7 accumulator to its own CSV;
  skips (does not fabricate) any accumulator left empty (e.g. `kd_loss_rows` if no requested method
  used KD).
- `write_r8_run_config_json()`: the Section 15 configuration truth table, as actually used
  (including any smoke-test caps, so a bounded run's config file is never silently
  indistinguishable from a full one).

All four writers are unit-tested against synthetic fake `results`/accumulator rows in a temporary
directory (`tempfile.mkdtemp()`), never touching CIFAR/CLIP, then cleaned up.

**Resume/incompatible-configuration risk**: there is no "resume" mechanism at all (by design,
matching R7, which also never resumes — every run trains fresh) — so there is no code path that
could mix an old run's partial state with a new one's config. The only way two runs' outputs could
collide is reusing the same `run_tag` twice, which `create_r8_run_directory()` explicitly refuses.

**One residual, low-severity note**: the Section 8.7 diagnostics accumulators are module-level
Python lists, cleared only by process restart. `--mode train`'s single one-shot invocation of
`run_full_r8_experiment()` per process (the only way the CLI calls it) means this is never actually
exercised as a problem for a real cluster job; it would only matter if `run_full_r8_experiment()`
were called a second time within the same live process (e.g. two `--mode train` invocations without
restarting the interpreter), which nothing in this file's CLI ever does.

**DATA-SPLIT STARTUP PERFORMANCE**: see Section 13. **Output/resume/failure: fixed and
self-tested.**

---

## 15. Configuration truth table

Statically verified against the module-level constants (all read directly, not inferred):

| Field | Value | Source |
|---|---|---|
| Dataset | CIFAR-100 | `DATASET_NAME = "cifar100"` |
| Protocol | 5×20 | `NUM_STEPS=5`, `CLASSES_PER_STEP=20` |
| Seed | 42 | `SEED = 42` |
| Epochs | 9 per step | `EPOCHS_PER_STEP = 9` |
| SimpleAvg | rank 80, alpha 160, scaling 2 | `LORA_R=80`, `LORA_ALPHA=160`, `LORA_SCALING=LORA_ALPHA/LORA_R=2.0` |
| RankExt | cumulative ranks [16,32,48,64,80], scaling 2 | `RANKEXT_RANK_SCHEDULE`, `RANKEXT_SCALING=RANKEXT_ALPHA_PER_RANK=2.0` |
| Targets | q_proj, v_proj | `TARGET_MODULES` |
| KD | old-seen only, T=2, T², weight=1, 1-epoch warmup | `KD_TEMPERATURE=2.0`, `KD_BASE_WEIGHT=1.0`, `KD_WARMUP_EPOCHS=1.0` |
| DenseOrth | whole-matrix dense-delta cosine², λ=20, 1-epoch warmup | `SELECTED_SHARED_LAMBDA=20.0`, `ORTH_WARMUP_EPOCHS=1.0` |
| Calibration | confidence_weighted_regime_grouped | `CALIBRATION_MODE` |
| ImageNet path | **none** | no reference to ImageNet/`imagenet` anywhere in this file |
| Device | CUDA if available, else CPU | `DEVICE` (fixed this pass) |
| Weight decay | 0.05, bias-exempt | `WEIGHT_DECAY = 0.05` (fixed this pass) |
| Optimizer / scheduler | AdamW / cosine | `OPTIMIZER_NAME`, `SCHEDULER_NAME`, `torch.optim.AdamW`/`CosineAnnealingLR` |
| Validation | 25/class held-out from TRAIN split | `VALIDATION_PER_CLASS = 25` |

---

## 16. Static comparison against R7

| Component | R7 | R8 | Verdict |
|---|---|---|---|
| Dataset | `load_dataset("cifar100")`, `fine_label` | identical | MATCH |
| Class order | contiguous `range(i*20,(i+1)*20)`, unshuffled | identical (`classes_for_step`) | MATCH |
| Train/val split | per-class shuffle(seed+cls), holdout 25/class | identical semantics, optimized implementation (Section 13) | MATCH (bit-for-bit, self-tested) |
| Optimizer | AdamW, decay/no-decay split, head-LR multiplier | now identical (fixed this pass — was previously missing the decay split and had wrong `WEIGHT_DECAY`) | **MATCH (after fix)** |
| LR | 5e-5 (SimpleAvg), 1e-4 (RankExt) | identical | MATCH |
| Weight decay | 0.05, bias-exempt | now identical (fixed this pass) | **MATCH (after fix)** |
| Scheduler | cosine | identical | MATCH |
| Epochs | 9/step | identical | MATCH |
| Batching | 16 | identical | MATCH |
| Best epoch | CE-only argmin, simple policy | identical | MATCH |
| Classifier | `nn.Linear(hidden,100)` (+ PEFT `modules_to_save` for SimpleAvg) | identical | MATCH |
| SimpleAvg merge | arithmetic mean of dense deltas + classifier stitching | identical | MATCH |
| RankExt schedule | [16,32,48,64,80], scaling=alpha_per_rank=2.0 | identical | MATCH |
| Calibration | confidence_weighted_regime_grouped, both families | identical | MATCH |
| Open evaluation | argmax over full 100-way logits | identical | MATCH |
| Restricted evaluation | same logits, eligible-class mask before argmax | identical | MATCH |
| Device placement | (implicit — R7 runs via HF `Trainer`, which manages device placement) | **was completely absent; fixed this pass** | **FIXED (was ERROR)** |
| KD support | full 100-way | **old-seen only** | INTENTIONAL CHANGE |
| KD warmup | none (SimpleAvg full-KD), none (RankExt full-KD) | 1-epoch warmup, both families | INTENTIONAL CHANGE |
| Orthogonality | factor-space (SimpleAvg mean-of-factors; RankExt frozen-block factor overlap) | dense-update-space cosine² | INTENTIONAL CHANGE |
| Orth warmup | asymmetric (SimpleAvg FO-alone: none; SimpleAvg FO+KD: 1 epoch; RankExt: always) | symmetric, 1 epoch, both families, all variants | INTENTIONAL CHANGE |
| Orth λ | 50 (factor-space) | 20 (dense-space, gradient-calibrated) | INTENTIONAL CHANGE |
| RankExt individual-block retention | R7 collapses history into one concatenated frozen blob | R8 keeps each block separately (`GrowingRankLoRALinearDenseOrth`) | INTENTIONAL CHANGE (necessary for DenseOrth's individual-reference requirement) |
| RankExt teacher-snapshot ordering | n/a (R7's KD teacher construction differs entirely) | new correctness requirement specific to R8's persistent-block design; implemented, audited, self-tested | INTENTIONAL CHANGE (necessary) |
| **RankExt new-block output warmup** | KD-conditional (disabled for KD methods) | **uniform (enabled for all 4) — DECISION 1, final** | **INTENTIONAL CHANGE (decided, documented)** |
| BWT / forgetting metrics | computed (family-specific methodology) | **implemented — DECISION 2, ported from R7's canonical formulas** | MATCH (formulas identical; RankExt evaluation-timing/asymmetry vs. calibration also matches R7 exactly) |

### Final list of intentional R8 changes (superseding the earlier, incomplete list)

1. Old-seen-only KD support (both families, identical mask formula).
2. Symmetric one-epoch KD weight warmup (both families, identical function).
3. Dense-delta whole-matrix cosine² DenseOrth (replaces factor-space orthogonality, both families).
4. Symmetric one-epoch DenseOrth λ warmup (both families, identical function — removes R7's
   SimpleAvg-FO-alone-vs-FO+KD and SimpleAvg-vs-RankExt asymmetries).
5. Shared DenseOrth λ = 20 (gradient-ratio-calibrated, not loss-magnitude-calibrated).
6. Individual RankExt frozen blocks retained separately (`GrowingRankLoRALinearDenseOrth`),
   required for DenseOrth's individual-previous-block reference (R7 collapses history into one
   concatenated blob, which cannot supply this).
7. Teacher snapshot taken *before* RankExt `grow()` mutates the live model in place — a new
   correctness requirement specific to R8's persistent-individual-block design, with no direct R7
   analogue to diverge from.
8. **Uniform RankExt new-block output warmup across all four R8 RankExt methods** (Decision 1,
   this pass) — R7's KD-conditional disable is *not* reproduced. **R8's KD RankExt methods
   therefore do NOT exactly match historical R7 KD variants in new-block warmup policy** — any
   R7-KD → R8-KD comparison carries this second difference alongside the intended KD-formula
   change, and must not be presented as a pure, single-variable historical comparison.
9. Implementation/performance changes that are provably scientifically equivalent: cached
   (single-pass) train/val split construction (Section 13, bit-for-bit membership-equivalent,
   self-tested against an independently-reimplemented reference) — not a scientific change at
   all, listed here only for completeness.

**Everything else found to differ from R7 (device placement, weight decay, the decay/no-decay
optimizer split, the RankExt step-1 grow bug, output persistence) was an implementation BUG in
R8, not an R7-vs-R8 design difference — all five are now fixed to MATCH R7 (or, for output
persistence and device placement, to make R8 functional at all, since R7's own mechanism for
those — HF `Trainer` — has no direct R8 analogue to diverge from).**

---

## 17. Verification (lightweight checks only — no CIFAR/CLIP/GPU/training executed)

```
$ python -m py_compile experiments_prepared/supervisor_regularizer_repair_both_families_r8.py
PYCOMPILE_OK

$ python experiments_prepared/supervisor_regularizer_repair_both_families_r8.py --mode selftest
R8 CODE-SAFETY SELF-TEST (106/106 passed)
OVERALL: PASS
```

This pass (resolving Decisions 1 and 2) added 20 further checks on top of the previous pass's 86
(51 → 86 → 106 across the two audit passes), covering: the RankExt step-1 double-grow regression
(reproduced pre-fix with a synthetic module, confirmed fixed post-fix, full 5-step rank table
verified), teacher immutability, DEVICE resolution/placement, batch-to-device movement in both
trainers, the data-split single-pass/reference-filter bit-for-bit equivalence, `WEIGHT_DECAY`'s
corrected value and bias-exemption split, removal of `build_optimizer_from_params`, every
output-persistence function against synthetic fake data — **plus, new this pass**: the uniform
RankExt new-block-warmup policy verified identical across all 4 methods with no per-method gating;
feature-anchor/projected-protection re-confirmed absent; exact hand-computed synthetic BWT
(−0.1575) and forgetting (0.2125) reference checks for RankExt; an exact hand-computed synthetic
BWT-surrogate check (−0.17) for SimpleAvg; `avg_forgetting`'s NaN-by-construction property for
SimpleAvg; `forward_transfer`'s continued absence as real output data (excluding the one
explanatory comment documenting that absence); and CSV/JSON metric-output-shape checks confirming
`backward_transfer`/`avg_forgetting` are present for all 8 methods, with SimpleAvg's serializing
as NaN and RankExt's as real numbers.

`git status --short` on `experiments_prepared/supervisor_exp1_cifar100_5x20_fixed_rankext.py` and
`thesis_writing/chapter_1/`, `_2/`, `_3/` shows no changes from this task.

---

## Summary of fixes made in this pass

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | RankExt step-1 double-`grow()` → guaranteed `AssertionError` crash | **Critical (crash)** | Fixed: gate `grow()` on `step_idx > 0` |
| 2 | No GPU device placement anywhere → silent CPU-only execution | **Critical (infeasible runtime)** | Fixed: `DEVICE` constant, `.to(DEVICE)` in `fresh_pretrained_model()`, device-matched `A_new`/`B_new` construction |
| 3 | Training-loop batches never moved to device → device-mismatch crash once fix #2 is in place | **Critical (crash)** | Fixed: `.to(device)` inside both `compute_loss()` methods |
| 4 | `WEIGHT_DECAY` wrong (0.0 vs R7's 0.05); no bias-decay exemption; RankExt used a separate, cruder optimizer builder | **Correctness (silent divergence from R7)** | Fixed: `WEIGHT_DECAY=0.05`, bias exemption in `build_optimizer()`, unified both families onto it, removed `build_optimizer_from_params` |
| 5 | `build_classwise_train_val_splits` did 100 full-table scans (~45 min overhead) | **Performance** | Fixed: single label-column pass + per-class `.select()`, bit-for-bit equivalence self-tested |
| 6 | No output persistence whatsoever | **Critical (unrecoverable cluster job)** | Fixed: run directory, incremental per-method CSV writing, completeness verification, diagnostic tables, run-config JSON |
| 7 | RankExt new-block output warmup applied uniformly instead of R7's KD-conditional disable | Confound (not a bug) | **RESOLVED — Decision 1: uniform ON, documented as intentional change (Section 3)** |
| 8 | BWT/forgetting metrics absent for both families | Gap (not a bug) | **RESOLVED — Decision 2: ported verbatim from R7's canonical formulas, implemented and self-tested (Section 12)** |

---

## Final verdict

Every **implementation bug** found by this audit (items 1–6 above) has been fixed and is covered
by a new, passing self-test. The two items that were not bugs (7–8) have now been explicitly
decided — Decision 1 (uniform RankExt new-block warmup, documented with the required
non-overclaiming caveat about R7-KD → R8-KD comparisons) and Decision 2 (BWT/forgetting ported
from R7's canonical formulas, not reinvented, and self-tested against hand-computed exact
synthetic references) — and both are implemented, self-tested, and documented in this report.

R8 is **statically audited and self-tested; full execution remains to be validated by the actual
R8 run.**
