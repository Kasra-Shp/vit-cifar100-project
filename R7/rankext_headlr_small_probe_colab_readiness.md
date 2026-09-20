# RankExt Head-LR Small Probe — Google Colab Preparation Readiness

No training was run for this task, on this machine or on Colab. This document covers only the
creation and static verification of the Colab-compatible probe files, derived from the existing
cluster probe without modifying it.

## 1. Files created

- `experiments_prepared/rankext_headlr_small_probe_colab.py` — Colab-adapted copy of the cluster
  probe.
- `experiments_prepared/rankext_headlr_small_probe_colab.ipynb` — a real, `nbformat`-validated
  Jupyter notebook (19 cells: 1 title + 9 markdown/code pairs), organized per spec Section 12
  (notebook JSON generation succeeded directly — no fallback to the `.py` + markdown alternative
  was needed).
- `R7/rankext_headlr_small_probe_colab_readiness.md` — this file.

## 2. Files NOT modified (MD5-verified byte-identical before/after this task)

| File | MD5 |
|---|---|
| `experiments_prepared/rankext_headlr_small_probe.py` | `556fd462f226e3f05c3107e3207db1a6` (unchanged) |
| `experiments_prepared/slurm/rankext_headlr_small_probe.sbatch` | `a706f643c960bbd186444c339ad76ec8` (unchanged) |
| `experiments_prepared/final_9method_5x20_performance_recovery.py` | `048331dac367c5b940b0680f04630eed` (unchanged) |
| `experiments_prepared/final_8method_5x20_refined.py` | `4673af2628f3d47473abf0ab26bec657` (unchanged) |

**PASS.**

## 3. Scientific equivalence to the cluster probe

`rankext_headlr_small_probe_colab.py` is a byte-for-byte copy of
`experiments_prepared/rankext_headlr_small_probe.py` (itself already audited against the canonical
job4971615 source in `R7/rankext_headlr_small_probe_readiness.md` — see that report for the full
20-hunk diff audit: method selection, `NUM_STEPS` 5→3, `RANKEXT_EPOCHS` 9→2, the LR-sweep env var,
and the fixed assertion locks) **plus exactly two localized, additive edits**:

1. A "COLAB ADAPTATION BLOCK" inserted between the existing `scipy` import try/except and the
   `# In[ ]:` cell marker that precedes the `SEED` assignment (i.e., before any dataset/model code
   runs). Contains: Colab-safe HF dataset cache env-var defaults (Section 5 below), the GPU-required
   hard-fail check (Section 6), and reading `COLAB_BATCH_SIZE` into a local variable.
2. The `BATCH_LORA = 16` line changed to
   `BATCH_LORA = int(COLAB_BATCH_SIZE) if COLAB_BATCH_SIZE else 8` (Section 7 below).

Nothing else differs. In particular, unchanged from the cluster probe:

- Historical pre-task-scale-calibration RankExt pipeline (row-norm calibration only).
- `METHODS_TO_RUN`: `simple_avg: False`, `rank_extension_factor_orth_lam50_new: True`,
  `rank_extension_fullkd_T2_protect30: False`, `rank_extension_factor_orth_lam50_fullkd_T2_protect30: True`
  — exactly the primary (combined) + secondary (FactorOrth50 alone) arms, SimpleAvg off.
- KD (T=2, weight=1.0), `protect_weight=30.0`, FactorOrth (`lambda=50`), classifier restoration,
  rank schedule, LoRA q/v targets, optimizer/scheduler, base LR (`LR_LORA`/`LR_RANKEXT`) — untouched.
- `SEED = int(os.environ.get("REPLICATION_SEED", "42"))` → 42 by default.
- `PROBE_NUM_STEPS = 3`, `PROBE_RANKEXT_EPOCHS = 2`.
- `RANKEXT_HEADLR_PROBE_MULTIPLIER = float(os.environ.get("RANKEXT_HEADLR_PROBE_MULTIPLIER", "1.0"))`
  — same env-var sweep mechanism, same historical-baseline default.
- `RUN_NAME_BASE` construction (LR-tag + seed, no `RUN_TAG` timestamp) — still guarantees each LR
  condition's checkpoint directory is distinct and deterministic; `BASE_OUTPUT_DIR` additionally
  embeds a fresh `RUN_TAG` timestamp per invocation (unchanged cluster-probe behavior), which is why
  the Colab driver's resume/skip logic (Section 9 below) globs on `RUN_NAME_BASE` rather than
  checking one fixed directory path.

## 4. Colab-specific differences (additive only)

| Aspect | Cluster probe | Colab probe |
|---|---|---|
| SLURM | Submitted via `sbatch`; loop over LR values is a bash `for` loop in the `.sbatch` file | No SLURM; the same LR loop is a Python driver (notebook Cell 7), invoking the script via `subprocess.run` once per LR value |
| Repo path | `/nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project` (hardcoded in the `.sbatch`, absent from the `.py`) | `/content/vit-cifar100-project` (cloned by notebook Cell 2; no path is hardcoded inside the `.py` itself, same as the cluster probe) |
| GPU | Assumed present (`#SBATCH --gres=gpu:a40:1`) | Explicitly checked and hard-failed on if absent (new code, Section 6) |
| Batch size | Fixed `BATCH_LORA = 16` | `COLAB_BATCH_SIZE` env var, default 8 (Section 7) |
| Dataset cache | Cluster's default HF cache location (unset, uses `~/.cache`) | `HF_HOME`/`HF_DATASETS_CACHE` default to `/content/data/...` via `os.environ.setdefault` (no-op if already set) |
| Resume | SLURM auto-requeue + `CHECKPOINTS_DIR` keyed only on `RUN_NAME_BASE` (unchanged, inherited) | Same underlying mechanism, plus a notebook-level glob-based skip-already-completed-LR check (Section 9) so re-running the notebook cell after a Colab disconnect doesn't repeat finished conditions |
| Drive backup | N/A | Optional, off by default (Section 10) |

## 5. GPU requirement

`rankext_headlr_small_probe_colab.py` calls `torch.cuda.is_available()` immediately after imports,
before any dataset download or model construction, and raises `RuntimeError` with an explicit
"re-run with a GPU runtime" message if it returns `False`. It never silently proceeds on CPU. On
success it prints `torch.__version__`, `torch.version.cuda`, the GPU name
(`torch.cuda.get_device_name(0)`), and total VRAM (`torch.cuda.get_device_properties(0).total_memory`).

Verified live: `grep -c "torch.cuda.is_available" rankext_headlr_small_probe_colab.py` → 4 hits (the
new hard-fail check plus the 3 pre-existing, unmodified `torch.cuda.is_available()` uses inherited
from the cluster probe, e.g. `USE_FP16 = torch.cuda.is_available()`).

## 6. Batch-size policy

`COLAB_BATCH_SIZE` env var overrides `BATCH_LORA` only — the single batch-size constant RankExt's
training path actually consumes (`BATCH_FT` is a full-fine-tune constant, unused by any
RankExt/KD/protect30/FactorOrth code path in this file; confirmed by the original script's own
structure — RankExt never calls a full-fine-tune trainer). No other hyperparameter (rank, LoRA
config, KD, protect30, FactorOrth lambda, task count, epoch count, or the LR sweep itself) is
touched by this override, per spec Section 6/13.

**Default: 8** (half the cluster's `BATCH_LORA = 16`). Reasoning, documented in-line in the script:
CLIP ViT-B/16 with LoRA adapters (rank ≤ 80 on q/v projections only) is a small trainable-parameter
footprint — the frozen backbone's forward pass dominates memory, LoRA + classifier-head gradients
add comparatively little. The cluster's `BATCH_LORA=16` ran without incident on A40 (48GB); a T4 has
16GB. 8 is a genuinely conservative halving for headroom on a shared/variable-overhead Colab
instance — **this is a documented judgment call, not an empirically-measured OOM threshold** (the
probe has never actually been run on any Colab GPU, since no training was executed in this task).
Configurable via `COLAB_BATCH_SIZE=16` (or higher on an L4/A100) to move closer to cluster parity.
On CUDA OOM, the in-script guidance is to lower `COLAB_BATCH_SIZE` and re-run — explicitly not to
change any scientific hyperparameter.

## 7. Data-path policy

**UPDATE (2026-09-20): see Section 15 — the dataset-loading call site was changed post-readiness**
(from the bare `"cifar100"` id to the namespaced `"uoft-cs/cifar100"` mirror) after this doc was
originally written, because the claim below ("unchanged call site") turned out to break on Colab's
actual library versions the first time this was really run. Section 15 has the full story; the rest
of this section otherwise still holds.

`load_dataset("cifar100")` (HuggingFace `datasets`) was originally left exactly as in the
cluster probe. `HF_HOME` and `HF_DATASETS_CACHE` are set via `os.environ.setdefault(...)` to
`/content/data/hf_home` and `/content/data/hf_datasets_cache` respectively, immediately after
imports — `setdefault` means this is a no-op if either variable is already set (e.g. by the user, or
on the cluster where these are never set, so cluster behavior is provably unaffected). This
redirects the download/cache location to a Colab-conventional path under `/content` without touching
the dataset-loading call itself.

## 8. LR-sweep mechanism (Colab)

Notebook Cell 7 implements spec Section 8 pattern A: a Python `for lr in [0.5, 1.0, 2.0]:` loop that
invokes `subprocess.run(["python", "-u", SCRIPT_PATH], env={..., "RANKEXT_HEADLR_PROBE_MULTIPLIER": str(lr)})`
once per value — subprocess isolation was chosen over import-and-reset because this is a
9700+-line module-level-execution script (importing it twice in one process would re-execute all
module-level code with stale global state from the first import; a subprocess gives each LR
condition a fully clean process). Each subprocess run writes to its own
`RUN_NAME_BASE`-tagged output directory (the script's own existing, unchanged mechanism — see
Section 3), so no cross-LR overwrite is possible.

## 9. Resume/skip behavior

Minimal, directory-glob-based (no new checkpoint framework, per spec Section 9's explicit
instruction not to build one): before launching a given LR value, Cell 7 globs for
`results/{RUN_NAME_BASE}_*/reports/final_9method_output_checklist.txt` and checks it contains
`"OVERALL PASS"`. If found, that LR condition is skipped and its existing output directory is used
directly. This means re-running Cell 7 after a Colab disconnect only re-runs LR conditions that
hadn't finished, not conditions already completed successfully.

## 10. Drive backup behavior

Off by default (`MOUNT_DRIVE = False` in Cell 4). If enabled, `google.colab.drive.mount` is called
once, and Cell 7 copies each newly-completed LR condition's full result directory to
`/content/drive/MyDrive/vit-cifar100-project-results/<run_dir_name>/` immediately after that
condition finishes (not only at the very end — so a mid-sweep disconnect still preserves whatever
completed so far). Cell 11 additionally copies the 3 final analysis outputs (report/summary/PNG) to
a `R7_probe_outputs/` subfolder there if Drive is mounted.

## 11. Static checks (spec Section 13) — all run for real, results below

| Check | Result |
|---|---|
| `python -m py_compile experiments_prepared/rankext_headlr_small_probe_colab.py` | **PASS** |
| No SLURM command in active (non-comment) code | **PASS** — 0 non-comment `sbatch`/`srun` hits |
| No `/nfsd` path in active (non-comment) code | **PASS** — 0 non-comment `/nfsd` hits |
| Task-scale calibration absent/inactive | **PASS** — all 4 occurrences of `PerTaskScaleCalibration`/`RankExtCalibratedModel`/`apply_rankext_task_scale_calibration` are inherited doc-comments stating its absence, not code defining/calling it (same count/nature as already audited for the cluster probe in `R7/rankext_headlr_small_probe_readiness.md`) |
| Old row-norm calibration active | **PASS** (unchanged call site, inherited) |
| Exactly the intended RankExt methods active (+ SimpleAvg off) | **PASS** — `"simple_avg": False`, `"rank_extension_factor_orth_lam50_new": True`, `"rank_extension_fullkd_T2_protect30": False`, `"rank_extension_factor_orth_lam50_fullkd_T2_protect30": True` |
| LR sweep exactly {0.5, 1.0, 2.0} | **PASS** — `LR_MULTIPLIERS = [0.5, 1.0, 2.0]` in notebook Cell 5, matching the cluster `.sbatch`'s `LR_MULTIPLIERS=(0.5 1.0 2.0)` |
| 3 tasks / 2 epochs / seed 42 | **PASS** — `PROBE_NUM_STEPS = 3`, `PROBE_RANKEXT_EPOCHS = 2`, `SEED` default `"42"` |
| Output directories LR-specific | **PASS** — `RUN_NAME_BASE` embeds `_PROBE_LR_TAG` (Section 3) |
| GPU required | **PASS** (Section 5) |
| Canonical cluster probe (and its `.sbatch`, and both upstream canonical scripts) untouched | **PASS** — MD5-verified (Section 2) |
| Notebook is valid, `nbformat`-parseable JSON | **PASS** — `nbformat.read(..., as_version=4)` + `nbformat.validate()` clean, 19 cells, normalized (cell IDs present) |

## 12. Notebook structure

`experiments_prepared/rankext_headlr_small_probe_colab.ipynb`, 19 cells (title + 9 markdown/code
pairs), following spec Section 12 with Cells 7–9 combined into one "run full small sweep" cell as
the spec explicitly permits:

1. Title/overview (markdown)
2. Cell 1 — Runtime/GPU check (`!nvidia-smi`)
3. Cell 2 — Clone or update repository (from `https://github.com/Kasra-Shp/vit-cifar100-project.git`,
   idempotent `git pull` if already cloned)
4. Cell 3 — Dependency inspection/install (checks stock-vs-missing packages; installs only
   `transformers`/`peft`/`datasets` if missing; never reinstalls torch/torchvision; notes
   `scikit-learn` is checked but not actually required — this probe has no sklearn dependency)
5. Cell 4 — Optional Google Drive mount (off by default)
6. Cell 5 — Colab config / paths / batch size (`COLAB_BATCH_SIZE`, `LR_MULTIPLIERS`)
7. Cell 6 — Static probe verification (py_compile + the grep checks from Section 11, run live)
8. Cell 7 — Run LR 0.5 / 1.0 / 2.0 (combined sweep driver with resume/skip, Sections 8–9)
9. Cell 10 — Collect/analyze results (reads each completed run's summary table, writes
   `R7/rankext_headlr_small_probe_{report.md,summary.csv,.png}`, all explicitly labeled SMOKE /
   DIAGNOSTIC ONLY; documents that the stability-label judgment calls (Section 11/12 of the
   investigation spec) still require human inspection of the step-boundary loss/CE tables, not just
   final accuracy — the analysis cell does not auto-assign PROMISING/NEUTRAL/UNSTABLE/etc.)
10. Cell 11 — Save/copy artifacts (lists local R7 outputs; copies to Drive if mounted)

## 13. Exact usage

1. Open `experiments_prepared/rankext_headlr_small_probe_colab.ipynb` in Google Colab (upload it, or
   open it directly from the cloned GitHub repo via Colab's "Open notebook → GitHub" if the file is
   pushed to the remote first).
2. Runtime → Change runtime type → GPU (T4 is Colab's free-tier default; L4/A100 also work).
3. Run Cell 1 and confirm `nvidia-smi` shows a GPU.
4. Run Cells 2–5 in order (clone, install deps, optional Drive mount, config).
5. Run Cell 6 (static verification) and confirm all checks PASS before proceeding.
6. Run Cell 7 (the 3-LR sweep). This is the only long-running cell.
7. Run Cell 10 (analysis) to produce the 3 `R7/rankext_headlr_small_probe_*` outputs.
8. Run Cell 11 to confirm/back up the outputs, then download them from the Colab file browser (or
   from Drive, if mounted) before the runtime recycles.

## 14. Rough runtime estimate (T4 / L4) — clearly a rough guess, not a measurement

**No real measurement exists for this probe on any hardware** — per the prior task's terminal
summary, the cluster probe itself has never actually been executed (`RESULTS: NOT RUN`), so there is
no measured probe runtime to scale from on Colab either. The only real cluster timing data point
available in this repo is job 4972616's *full* 8-method benchmark
(`R7/cifar100_5x20_final_8method_refined_4972616/.../`, directory-timestamp span 2026-09-19 19:05:36
→ 2026-09-20 02:54, i.e. **≈7h49m** on an A40, for 8 methods × 5 steps × 9 epochs = 360
method-step-epochs of work).

Scaling this probe's workload (2 RankExt methods × 3 steps × 2 epochs = 12 method-step-epochs, ≈1/30
of that count) against the same reference run gives a naive proportional estimate of ~16 min/LR
condition on an A40. RankExt (especially the combined KD+protect30+FactorOrth arm) is heavier
per-epoch than the SimpleAvg arms that make up half of the reference run's 360-unit denominator, and
there's fixed per-process startup/data-loading overhead that doesn't shrink with epoch count, so a
more realistic A40-equivalent estimate is **≈20–30 min per LR condition, ≈1–1.5h total for the
3-value sweep**.

Applying commonly-cited relative FP16/BF16 training-throughput ratios (T4 roughly 3–4× slower than
A40 for ViT-scale transformer training; L4 roughly comparable to, sometimes modestly slower than,
A40 depending on precision/kernel support) gives:

- **T4**: roughly **1–2 hours per LR condition, ≈3–6 hours total** for the 3-value sweep.
- **L4**: roughly **25–40 min per LR condition, ≈1.5–2h total** for the 3-value sweep.

These are order-of-magnitude planning estimates only, clearly labeled as such — treat the actual
first Colab run as the real measurement, and budget a T4 session (which can disconnect after
extended idle/runtime limits) accordingly; the resume/skip logic (Section 9) exists specifically to
make a multi-session T4 run tractable if a single session isn't long enough.

---

## 15. POST-READINESS CORRECTIONS (2026-09-20) — found via the first real Colab execution

This readiness doc's static checks (Section 11) passed, but this was still the first time the
Colab-adapted script was actually executed on a real GPU, and it surfaced two real bugs the static
checks could not have caught (compile/grep/MD5 checks don't execute the training-driver code paths):

**Bug 1 — dataset loading (`HfUriError`).** `load_dataset("cifar100")` failed on Colab's current
`datasets`/`huggingface_hub` versions: their stricter HF-URI parser rejects the legacy un-namespaced
single-segment repo id `"cifar100"` when resolving its revision-pinned config file
(`HfUriError: Repository id must be 'namespace/name', got 'cifar100'`). This does not happen on the
cluster (older, pinned library versions there), so the cluster probe was correctly left unchanged.
**Fix:** switched to `"uoft-cs/cifar100"`, the actively-maintained namespaced Hub mirror — verified
(via a live fetch of its dataset card) to have an identical schema (`img`/`fine_label`/`coarse_label`
columns, train=50000/test=10000 rows), matching exactly what this script's own dataset-identity
assertions require, so no downstream code changes were needed. A fallback to the bare `"cifar100"`
id is kept in case the namespaced mirror is ever unavailable. Empirically confirmed working: the
next run's stdout showed `Loaded dataset via namespaced Colab-safe mirror: uoft-cs/cifar100` followed
by `EXPERIMENT 1 dataset identity check PASSED`.

**Bug 2 — rank-schedule length mismatch (guaranteed crash, not Colab-specific).**
`get_rank_extension_rank_schedule()` required `len(schedule) == NUM_STEPS`, but
`active_rankext_rank_schedule()` always returns the full canonical 5-entry `[16, 32, 48, 64, 80]`
list (correctly — hard assertions elsewhere require this), while this probe sets
`NUM_STEPS = PROBE_NUM_STEPS = 3`. This mismatch was unconditional: it crashed the very first
RankExt arm's training on the very first LR condition, with a real traceback:
`ValueError: active rank schedule must have NUM_STEPS=3 entries, got [16, 32, 48, 64, 80]`.
**This bug is identical in the cluster probe** (`experiments_prepared/rankext_headlr_small_probe.py`,
same function/lines) — it was not Colab-specific, just first discovered here because this was the
first environment where the script's training path actually executed. **Fix:** in both this Colab
script and the cluster probe, `get_rank_extension_rank_schedule()` now slices the canonical schedule
to `schedule[:PROBE_NUM_STEPS]` before the length check, preserving the exact same per-step rank
values (step 1→16, step 2→32, step 3→48) as the full run — no rank/LoRA/KD/protect30/FactorOrth
hyperparameter changed, only how many of the unchanged canonical schedule's entries this reduced
probe consumes. See `R7/rankext_headlr_small_probe_readiness.md` Section 13a for the cluster-side
fix record (MD5/compile-reverified there too).

**Verification status of both fixes:**
- Colab script: **Bug 1 empirically confirmed fixed** (real Colab T4 run, log excerpt above); Bug 2
  fixed in the same file but not yet re-run end-to-end past that point at the time of this update
  (the user's next Colab run will confirm it).
- Cluster probe: both-equivalent fix applied and statically re-verified (`py_compile`,
  `bash -n` on the launcher) — **not yet confirmed by an actual cluster run**, since this
  environment still has no cluster/GPU access (Section 5/12 of the cluster readiness doc).

Practical implication: the sweep may still hit further real-execution bugs neither static check
could catch (this is the nature of a first real run of new reduced-scale code) — report any new
traceback and it will be triaged the same way.

# FINAL TERMINAL SUMMARY

```
RANKEXT HEAD-LR COLAB PREPARATION

SOURCE
Cluster probe: experiments_prepared/rankext_headlr_small_probe.py (unmodified)
Colab script: experiments_prepared/rankext_headlr_small_probe_colab.py
Colab notebook: experiments_prepared/rankext_headlr_small_probe_colab.ipynb
Canonical cluster probe modified: NO

SCIENTIFIC CONFIG
New task-scale calibration active: NO (absent from source; inherited from cluster probe)
Old row-norm calibration active: YES (unchanged)
Methods: rank_extension_factor_orth_lam50_fullkd_T2_protect30 (primary), rank_extension_factor_orth_lam50 (secondary); SimpleAvg OFF
LR multipliers: 0.5, 1.0 (historical baseline), 2.0
Tasks: 3 of 5 (Task1, Task1->2, Task2->3)
Epochs: 2/step
Seed: 42

COLAB
SLURM dependency: NONE (Python subprocess loop replaces the sbatch bash loop)
Cluster paths: NONE in the .py (repo path handled by notebook Cell 2's clone/cd, not hardcoded)
GPU required: YES (hard-fail if torch.cuda.is_available() is False)
Default batch size: 8 (COLAB_BATCH_SIZE env var, conservative default for 16GB T4)
Batch size configurable: YES (COLAB_BATCH_SIZE env var)
Drive optional: YES (off by default, Cell 4)
Resume support: YES (glob-based skip of already-completed LR conditions, Cell 7)

STATIC CHECKS
Python compile: PASS
No /nfsd active path: PASS
No sbatch/srun dependency: PASS
Output directories LR-specific: PASS
Canonical source unchanged: PASS (MD5-verified, 4 files)

FILES
Colab script: experiments_prepared/rankext_headlr_small_probe_colab.py
Colab notebook: experiments_prepared/rankext_headlr_small_probe_colab.ipynb
Readiness report: R7/rankext_headlr_small_probe_colab_readiness.md

TRAINING EXECUTED LOCALLY: NO
READY FOR GOOGLE COLAB: YES
```
