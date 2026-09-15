# Supervisor Experiment 2 -- Colab/Cluster Equivalence Audit

**Date:** 2026-09-15
**Status:** PREPARED, NOT LAUNCHED (no training run in preparing this notebook)

## Summary

Supervisor Experiment 2 (already audited and finalized -- see
`thesis_agent/reports/supervisor_exp2_preparation_audit.md`) is now runnable
on Google Colab via a thin notebook runner that executes the **same,
unmodified-in-substance** script the cluster uses. No scientifically separate
implementation was created. Two small pieces of implementation-only
infrastructure were added directly to the shared script (so cluster and
Colab never diverge): environment-variable-overridable output paths, and
method-level result persistence/resume. Both are strict no-ops on the
cluster unless a job explicitly opts in via the same environment variables.

## Files

| | Path |
|---|---|
| Scientific source of truth (edited only for infra, see below) | `experiments_prepared/supervisor_exp2_cifar100_5x20_rank32_vs_rankext160.py` |
| Colab notebook (new) | `colab/supervisor_exp2_cifar100_5x20_rank32_vs_rankext160_colab.ipynb` |

## What changed in the script, and why

A byte-level `diff` against the pre-Colab-prep version of the script (the
version that already passed the preparation audit) shows exactly two new
change regions, both purely infrastructural:

### 1. `ROOT_RESULTS_DIR` / `RUN_TAG` made environment-overridable

```python
ROOT_RESULTS_DIR = os.environ.get("EXP2_RESULTS_ROOT", "results")
RUN_TAG = os.environ.get("EXP2_RUN_TAG") or datetime.now().strftime("%Y%m%d_%H%M%S")
```

- Unset (the only behavior any cluster job has today): **identical** to
  before -- relative `"results"` directory, a fresh timestamp every launch.
- The Colab notebook sets both: `EXP2_RESULTS_ROOT` to a persistent Google
  Drive path, and `EXP2_RUN_TAG` to a tag that is written once to Drive and
  reused across reconnects, so `BASE_OUTPUT_DIR` (and everything under it)
  lands at the same path every time this logical experiment attempt is
  resumed.
- `METHOD_RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "method_results")` was
  added alongside the other `*_DIR` constants (`TABLES_DIR`, `PLOTS_DIR`,
  etc.), created the same way (`os.makedirs(..., exist_ok=True)`).

Nothing about *what* gets computed changes -- only *where* it is written.

### 2. Method-level result persistence + resume

**Audit of the pre-existing script found results are persisted only ONCE,
after ALL 8 methods finish** (`method_summary_df.to_csv(...)` and every
other final table are built from module-global Python lists -- e.g.
`method_summary_rows`, `epoch_loss_rows`, `best_epoch_selection_rows` -- that
are populated in-memory across both the `simple_avg_execution_order` and
`rank_extension_execution_order` loops and never touch disk until after both
loops complete). No per-method checkpoint or result file existed before this
change; a Colab disconnect at any point would have lost every method trained
so far, in full.

**Fix implemented (method-level only, per instruction -- no batch/epoch
resume):**

- ~13 module-global accumulator lists/dicts (`all_results`,
  `method_summary_rows`, `train_diagnostic_rows`, `epoch_loss_rows`,
  `best_epoch_selection_rows`, `per_step_accuracy_rows`,
  `per_step_accuracy_restricted_rows`, `classifier_row_norm_diagnostic_rows`,
  `classifier_confidence_calibration_diagnostic_rows`,
  `rankext_bias_diagnostic_rows`, `rankext_feature_alignment_diagnostic_rows`,
  `rankext_new_block_warmup_diagnostic_rows`, `orth_kd_eval_rows`, and the
  dict `rank_extension_stepwise_accuracy_by_method`) were identified by
  reading every `.append()`/`.extend()` call site feeding a final CSV/table.
  Each is populated **strictly within one method's own
  `run_simple_avg_variant()` / `run_rank_extension_variant()` call** (verified
  -- nothing touches them between methods, and both execution-order loops
  call exactly one method at a time, sequentially).
- A generic snapshot-length-before / diff-after wrapper
  (`_resume_run_method_or_skip`) around each method call in both loops:
  - Before training a method: checks for an existing, complete,
    config-fingerprint-matched `method_results/<method>.json`. If found,
    **skips training** and replays its persisted rows back into the same
    accumulator lists/dict instead.
  - If not found (or invalid/stale): calls the method's existing
    `run_*_variant()` function **completely unchanged**, then persists
    exactly the rows that call newly contributed, atomically (`write to
    .tmp` -> `os.replace()`, same-directory rename -- a crash mid-write
    leaves only an ignored `.tmp` file, never a half-written final file).
  - A config fingerprint (seed, protocol, SimpleAvg rank/alpha, RankExt
    schedule, target modules, KD/FactorOrth params, method-name set) is
    stored with every persisted result and re-checked on load; a mismatch
    (e.g. a persisted file from a differently-configured run reusing the
    same directory) is treated as **not completed**, never silently reused.
  - A per-process guard (`_RESUME_METHODS_LOADED_THIS_PROCESS`) prevents
    double-counting rows if the same method were ever re-entered within one
    process (e.g. a notebook cell re-run without a runtime restart) -- not
    needed for the normal one-call-per-method path, pure safety net.
- On a **fresh run** (no persisted files -- the only case any cluster job
  hits today, and the case any brand-new Colab run also hits), this
  mechanism is a pure pass-through: `_resume_try_load_method_result` returns
  `None`, the method's own `run_*_variant()` is called exactly as before,
  with no change to `set_seed()`, model/optimizer construction, data
  ordering, or any method's training math. **The only new behavior is that a
  small JSON file is additionally written to `method_results/` after each
  method finishes.**

This satisfies the instruction to keep resume "implementation-only, not
scientific," and to make it "identical infrastructure whether launched from
Colab or from the cluster" -- it is literally the same code path, gated
purely by whether a stale/absent file is found.

## Cluster-specific paths

Grepped the script for `/nfsd` (the UniPD-cluster-only prefix named in the
request) and any other absolute path literal: **none found**. The script
already used only relative paths (`"results"`, now env-overridable) and
Hugging Face Hub identifiers (`"openai/clip-vit-base-patch16"`,
`"cifar100""`), never a filesystem path baked in. No path rewriting was
needed beyond the `ROOT_RESULTS_DIR`/cache env-var support already described
-- both explicitly authorized as "environment variables ... if needed" by
the request, and both optional / additive.

## Static difference audit: cluster Exp2 vs. Colab Exp2

| Component | Cluster Exp2 | Colab Exp2 | Verdict |
|---|---|---|---|
| Dataset | CIFAR-100 | CIFAR-100 | MATCH |
| Protocol | 5x20 | 5x20 | MATCH |
| Seed | 42 | 42 (REPLICATION_SEED explicitly unset in the launch cell, matching the sbatch's own `unset REPLICATION_SEED` convention) | MATCH |
| Epochs | 9 | 9 | MATCH |
| Methods | 8 canonical (no R8) | 8 canonical (no R8) | MATCH |
| SimpleAvg rank/alpha/scaling | 32 / 64 / 2.0 | 32 / 64 / 2.0 (unedited) | MATCH |
| RankExt schedule/scaling | [32,64,96,128,160] / 2.0 | [32,64,96,128,160] / 2.0 (unedited) | MATCH |
| KD | T=2, weight=1, current-step-only | unedited | MATCH |
| FactorOrth | lambda=50, Exp1 warmup | unedited | MATCH |
| Optimizer | AdamW | unedited | MATCH |
| Learning rate | LR_LORA=5e-5, LR_RANKEXT=1e-4 | unedited | MATCH |
| Weight decay | 0.05 | unedited | MATCH |
| Scheduler | cosine | unedited | MATCH |
| Batch size | BATCH_LORA=16 | unedited | MATCH |
| Best-epoch selection | validation CE | unedited | MATCH |
| Merge (SimpleAvg) | dense-delta arithmetic mean | unedited | MATCH |
| Classifier handling | per-step stitched / masked-protected | unedited | MATCH |
| Calibration | confidence_weighted_regime_grouped | unedited | MATCH |
| Metrics (open/restricted/BWT/forgetting) | unedited | unedited | MATCH |
| Mixed precision | `fp16=USE_FP16` (already present, `USE_FP16 = torch.cuda.is_available()`) | same, unedited -- applies automatically on Colab's GPU | MATCH |
| Output directory | `results/<run>_<timestamp>/` (relative to CWD) | `$EXP2_RESULTS_ROOT/<run>_<tag>/` (Drive, persistent) | **INFRASTRUCTURE-ONLY** (env-var override, default unchanged) |
| Run tag / resume | fresh timestamp per launch (no resume) | stable tag reused across reconnects, enabling method-level resume | **INFRASTRUCTURE-ONLY** (env-var override, default unchanged; resume logic itself ships in the same script and is a no-op with nothing persisted) |
| HF/dataset cache location | default (`~/.cache/huggingface`) | `/content/hf_cache` by default (or Drive, if the user uncomments that option) | **INFRASTRUCTURE-ONLY** (standard HF env vars only; zero script changes) |
| Library versions (torch/transformers/peft/datasets) | whatever the cluster's environment has | installed/verified fresh in the Colab session; torch/torchvision never touched | **INFRASTRUCTURE, DOCUMENTED CAVEAT** -- see below |

Every scientific row is **MATCH**. All differences are infrastructure-only
and are all documented above and in the notebook itself.

### Documented caveat: library versions are not pinned

No `requirements.txt` (or `environment.yml`, or any pinned dependency
manifest) exists anywhere in this repository -- checked directly, not
assumed. This means there was never a single canonical dependency-version
set to reproduce in the first place; the cluster's own installed
`transformers`/`peft`/`datasets` versions are whatever that environment
happens to have, undocumented. The Colab notebook installs current versions
of `transformers`, `datasets`, `peft` (and optionally `scipy`) only if
missing, and immediately verifies every import the script needs succeeds.
This is the same caveat that would apply between any two non-identical
Python environments running this script (including two different cluster
nodes) -- not something this Colab preparation introduces. `torch` and
`torchvision` are never installed/upgraded by the notebook, preserving
Colab's own CUDA-matched build.

## Method-level resume: validation performed

A lightweight synthetic test (`thesis_agent`-adjacent, not committed -- run
directly against the exact resume-infrastructure code block extracted
verbatim from the script) exercised the mechanism with fake methods and fake
row data (no CLIP/CIFAR involved), covering every case the instructions
asked for:

- **Fresh method trains exactly once, result persisted to a real
  `method_results/<method>.json` file** -- PASS
- **Re-running detects the completed method and skips retraining**, with
  accumulator lists byte-identical (no duplication, no corruption) after the
  skip, including the `rank_extension_stepwise_accuracy_by_method` dict
  entry -- PASS
- **A partially-written result (`.tmp` file, or a final file present but
  missing `status: "complete"`) is never mistaken for completed** -- the
  method still trains -- PASS
- **A persisted result produced under a different scientific configuration
  (different seed, simulating a stale/reused directory) is detected via the
  config fingerprint and ignored -- the method retrains rather than reusing
  a stale/mismatched result** -- PASS
- **Config fingerprint is deterministic** given identical config -- PASS

All checks passed after one fix during testing: the first implementation
would double-append a persisted result's rows if the same method were
resumed twice within one process; a per-process guard set
(`_RESUME_METHODS_LOADED_THIS_PROCESS`) was added to fix this (see above) --
this does not affect the normal one-call-per-method-per-process path at all,
only the edge case the test surfaced.

Final aggregation (`method_summary_df = pd.DataFrame(method_summary_rows)`
and every other post-loop table/report) is completely unmodified and still
runs unconditionally after both loops -- since resume replays into the exact
same accumulator objects that code already reads, a mixed
trained/resumed/8-method run produces output indistinguishable from an
uninterrupted single-process run. The existing `ACTIVE_METHOD_NAMES`
exact-8-method assert (unmodified, from the preparation-audit hard-assertion
block) still gates the whole run regardless of which methods were resumed
vs. freshly trained.

## GPU memory risk (static assessment)

CLIP-ViT-B/16 (~86M frozen backbone params) with LoRA on `q_proj`/`v_proj`
only, up to RankExt's final cumulative rank 160: LoRA parameter counts at
this width are on the order of a few million, tiny relative to the frozen
backbone. Batch size is 16 (unchanged), training already runs under
`fp16=True` whenever CUDA is available (pre-existing, unedited --
`USE_FP16 = torch.cuda.is_available()`, wired into every `TrainingArguments`
call), and every method explicitly frees GPU memory before the next one
starts (`del <model>; gc.collect(); torch.cuda.empty_cache()`, pre-existing,
unedited). **Assessed risk: LOW** on any Colab GPU tier (T4/16GB and above).
No rank, batch size, model, precision, or method count was reduced to reach
this conclusion, and no AMP/precision change was introduced (fp16 was
already there; nothing new was added on top of it).

## Validation performed

1. `python -m py_compile experiments_prepared/supervisor_exp2_cifar100_5x20_rank32_vs_rankext160.py` -- **PASS**
2. Full config/assertion pipeline dry-run (everything up to, not including,
   dataset load -- zero data download, zero model build, zero training),
   including the new `EXP2_RESULTS_ROOT`/`EXP2_RUN_TAG` overrides and the
   resume-infrastructure definitions -- **PASS**, `method_results/` directory
   confirmed created under the overridden results root.
3. `nbformat.validate()` on the notebook -- **PASS**
4. Every notebook code cell's source `compile()`d for Python syntax
   (all cells use plain `subprocess`/`os` calls rather than IPython shell
   magics, so this is a direct, complete syntax check, not an approximation)
   -- **PASS**, all 9 code cells valid.
5. Synthetic resume-infrastructure tests (see above) -- **ALL PASSED**.
6. Byte-level `diff` against the pre-Colab-prep script confirmed the only
   changed regions are exactly the two described above (`ROOT_RESULTS_DIR`/
   `RUN_TAG`/`METHOD_RESULTS_DIR` block, and the resume-infrastructure
   block + its two call sites) -- no scientific code path touched.

No dataset was downloaded, no CLIP model was built, and no training was run
at any point while preparing this notebook.
