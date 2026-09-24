# ImageNet-100 shared-cluster setup and UniPD production workflow

The preferred production source is the ImageNet-1K directory already mounted on UniPD:

```text
/nfsd/lttm4/datasets/ImageNet-1k_torch/
  train/<WNID>/...
  val/<WNID>/...
```

The project’s fixed 100 WNIDs in `tools/imagenet100_common.py` and the checked-in seed-42 class-order artifact are the only benchmark definition. The loader reads only those 100 directories, ignores the other 900 classes, creates deterministic calibration/test subsets from `val/`, and never copies or downloads image data.

## Preferred cluster path: no download required

Verify the shared root before submission:

```bash
python tools/verify_imagenet100_dataset.py \
  --data-root /nfsd/lttm4/datasets/ImageNet-1k_torch \
  --source-mode shared_imagenet1k \
  --max-readable-checks 24
```

Expected source mode is `SHARED_IMAGENET1K_FILTERED`, with 1000 source WNID directories and an exact 100/100 project-subset match. The validation policy is seed 42, 25 images/class for calibration, and the remaining selected validation images as frozen test. Exact paths and stable sample IDs are saved by production in `logs/shared_imagenet1k_source_manifest.jsonl`.

Submit the benchmark without any dataset preparation step:

```bash
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
export IMAGENET100_ROOT=/nfsd/lttm4/datasets/ImageNet-1k_torch
sbatch experiments_prepared/slurm/final_8method_imagenet100_5x20_job4971615_canonical_ep9.sbatch
```

The Slurm preflight prints the selected train/validation counts, all five groups of 20 WNIDs, the eight methods, seed 42, and nine epochs. It performs no download, extraction, or copying, and the cluster dataset path has no dataset network access.

## Historical public-source audit

The previously inspected public source was [clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100). Its confirmed schema is:

- splits: `train` and `validation`;
- fields: `image` and `label`;
- labels: a 100-entry Hugging Face `ClassLabel` in the CMC random-100 order;
- source counts: 126,689 train and 5,000 validation examples.

That public 100-class set is not the canonical thesis set: the exact audit is `INTERSECTION: 8`, with 92 missing and 92 extra WNIDs. Therefore the downloader refuses to materialize it as a canonical prepared directory. This is intentional; do not remap or substitute classes silently.

Run the cheap, streaming-only audit first:

```bash
python tools/download_imagenet100_external.py \
  --source hf \
  --dataset-name clane9/imagenet-100 \
  --output-dir D:/datasets/imagenet100_prepared \
  --seed 42 \
  --inspect-only
```

The command requires no `huggingface-cli login` and does not materialize image shards. The same command without `--inspect-only` remains the documented public preparation command, but it will stop with the same class-set failure until a public source matching the canonical WNIDs is identified:

```bash
python tools/download_imagenet100_external.py --source hf --dataset-name clane9/imagenet-100 --output-dir D:/datasets/imagenet100_prepared --seed 42
```

## Optional external preparation (not the cluster production path)

If the shared cluster root is unavailable, a licensed local ImageNet-1K tree can still be prepared offline. The gated option requires normal access approval and authentication; never put a token in a command or repository file:

```bash
huggingface-cli login
python tools/download_imagenet100_external.py \
  --source hf \
  --dataset-name ILSVRC/imagenet-1k \
  --cache-dir D:/datasets/hf-cache \
  --output-dir D:/datasets/imagenet100_prepared \
  --seed 42
```

Alternatively, use an existing licensed ImageNet-1k directory without network access:

```bash
python tools/download_imagenet100_external.py \
  --source local \
  --source-dir D:/datasets/ImageNet-1k \
  --output-dir D:/datasets/imagenet100_prepared \
  --seed 42
```

The local source must contain `train/<synset>/...` and `val/<synset>/...` for all canonical WNIDs. The default held-out policy is 25 calibration images/class and all remaining held-out images as frozen test images. HF caches are reusable/resumable and are never copied into the prepared output.

## Local prepared-layout verification

```bash
python tools/verify_imagenet100_dataset.py --data-root D:/datasets/imagenet100_prepared
```

This validates an already materialized prepared layout. For the cluster tree, use the shared-root command above.

## Packaging and checksum

```bash
python tools/package_imagenet100_for_cluster.py \
  --data-root D:/datasets/imagenet100_prepared \
  --output D:/datasets/imagenet100_prepared.tar.gz
```

The archive contains only `train/`, `calibration/`, `test/`, and `metadata/` under `imagenet100_prepared/`; HF caches and temporary files are excluded.

```bash
sha256sum imagenet100_prepared.tar.gz
```

PowerShell:

```powershell
Get-FileHash D:/datasets/imagenet100_prepared.tar.gz -Algorithm SHA256
```

Record the real archive checksum in `R8/imagenet100_dataset_source_audit.md` after packaging.

## Upload to the cluster

```bash
rsync -avP imagenet100_prepared.tar.gz shahrampou@login:<cluster-target>/
```

or:

```bash
scp imagenet100_prepared.tar.gz shahrampou@login:<cluster-target>/
```

Do not hard-code passwords.

## Output and CPU analysis artifacts

The production run writes timestamped results under `results/imagenet100_5x20_final_8method_canonical_job4971615_seed42_ep9_EPOCH9_MAIN_<timestamp>/` and checkpoints under the matching `_checkpoints` directory. Each method saves validation/test logits, labels, task IDs, sample IDs, classifier weights, and biases for later CPU-only calibration analysis, together with the shared-source manifest.
