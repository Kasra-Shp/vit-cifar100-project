# ImageNet-100 methods 6–8 continuation

Launcher: `final_8method_imagenet100_5x20_continuation_methods6to8_seed42_ep9.py`

Slurm submission:

```bash
sbatch experiments_prepared/slurm/final_8method_imagenet100_5x20_continuation_methods6to8_seed42_ep9.sbatch
```

The launcher writes results below
`results/imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_*`
and checkpoints below
`results/imagenet100_5x20_final_8method_canonical_continuation_methods6to8_seed42_ep9_checkpoints`.
Each RankExt checkpoint is overwritten atomically at a completed task boundary;
the payload records `completed_task_index`, so a later launch resumes at the
next task. A `DONE.marker` is created only after final evaluation succeeds.

After both runs finish, merge only the audited rows:

```bash
python tools/merge_imagenet100_continuation_results.py \
  --original-result <original-job-4991055-result-directory> \
  --continuation-result <continuation-EPOCH9_MAIN-result-directory> \
  --output <final-8method-summary.csv>
```

If methods 6--7 are already complete and Method 8 is recovered by the
dedicated launcher, pass the two continuation result roots separately:

```bash
python tools/merge_imagenet100_continuation_results.py \
  --original-result <original-job-4991055-result-directory> \
  --continuation-result <job-4994352-methods-6-and-7-result-directory> \
  --resumed-method8-result <method-8-resume-result-directory> \
  --output <final-8method-summary.csv>
```

The Method-8 launcher is
`final_8method_imagenet100_5x20_method8_resume.py` and its Slurm wrapper is
`slurm/final_8method_imagenet100_5x20_method8_resume.sbatch`. It requires the
exact Task-3 boundary checkpoint, disables methods 6 and 7 before active
configuration construction, restarts Task 4 at epoch 0, and creates its
`DONE.marker` only after Task 5 and final evaluation succeed.

If the original result has a run-specific `configs/protocol_manifest.json`,
pass it with `--original-protocol-manifest`; otherwise the checked-in canonical
manifest is used. The utility refuses missing/duplicate method rows, protocol
mismatches, schema mismatches, and fabricated rows.
