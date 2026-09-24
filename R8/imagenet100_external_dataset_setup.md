# ImageNet-100 external preparation and UniPD transfer workflow

This workflow prepares the canonical thesis ImageNet-100 subset: the first 100 ImageNet WNIDs in sorted order, `n01440764` through `n01855672`. The cluster receives ordinary JPEG files and JSON metadata; it never accesses Hugging Face or the network for dataset files.

## Public no-login source audit (default first step)

The preferred public source is [clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100). Its confirmed schema is:

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

## Canonical data preparation

For the current canonical benchmark, use a licensed/local ImageNet-1k directory or the gated HF mirror. The gated option requires normal access approval and authentication; never put a token in a command or repository file:

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

## Local verification

```bash
python tools/verify_imagenet100_dataset.py --data-root D:/datasets/imagenet100_prepared
```

This is offline. It validates the exact 100 WNIDs, seed-42 5x20 task artifact, all three splits, readable images, manifest identity, and split disjointness.

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

## Cluster extraction and verification

```bash
mkdir -p /nfsd/lttm4/tesisti/shahrampour/datasets
tar -xzf imagenet100_prepared.tar.gz -C /nfsd/lttm4/tesisti/shahrampour/datasets
export IMAGENET100_ROOT=/nfsd/lttm4/tesisti/shahrampour/datasets/imagenet100_prepared
python tools/verify_imagenet100_dataset.py --data-root "$IMAGENET100_ROOT"
```

The dataset stays outside Git. The verifier must pass before any Slurm submission.

## Benchmark submission (not executed here)

```bash
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
export IMAGENET100_ROOT=/nfsd/lttm4/tesisti/shahrampour/datasets/imagenet100_prepared
sbatch experiments_prepared/slurm/final_8method_imagenet100_5x20_job4971615_canonical_ep9.sbatch
```

The wrapper verifies the dataset before training. This command was not executed in the preparation pass.

## Output and CPU analysis artifacts

The production run writes timestamped results under `results/imagenet100_5x20_final_8method_canonical_job4971615_seed42_ep9_EPOCH9_MAIN_<timestamp>/` and checkpoints under the matching `_checkpoints` directory. Each method saves validation/test logits, labels, task IDs, sample IDs, classifier weights, and biases for later CPU-only calibration analysis.
