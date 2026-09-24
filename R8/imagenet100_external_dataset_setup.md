# ImageNet-100 external preparation and UniPD transfer workflow

This workflow prepares the repository's previously selected PODNet/DER/DyTox-compatible ImageNet-100 subset: the first 100 ImageNet WNIDs in sorted order, `n01440764` through `n01855672`. It does not use the unrelated `clane9/imagenet-100` class definition.

The selected source is the gated Hugging Face `ILSVRC/imagenet-1k` repository. Preparation is local/off-cluster only. The cluster receives ordinary JPEG folders and JSON metadata; training has no Hub or network dataset path.

## Local download

Use a machine with enough disk for the selected images and the temporary HF cache. The downloader supports the authenticated HF source:

```bash
huggingface-cli login
python tools/download_imagenet100_external.py \
  --source hf \
  --dataset-name ILSVRC/imagenet-1k \
  --cache-dir D:/datasets/hf-cache \
  --output-dir D:/datasets/imagenet100_prepared
```

Do not put a token in a command committed to the repository. `HF_TOKEN` is also accepted locally. If ImageNet access has not been granted, the downloader fails with an access message; it does not bypass authentication or terms.

If an existing licensed ImageNet-1k directory is available locally, use the no-network source instead:

```bash
python tools/download_imagenet100_external.py \
  --source local \
  --source-dir D:/datasets/ImageNet-1k \
  --output-dir D:/datasets/imagenet100_prepared
```

The local source must contain `train/<synset>/...` and `val/<synset>/...` for the 100 selected WNIDs. The default held-out policy is 25 calibration images/class and all remaining held-out images as frozen test images.

## Local verification

```bash
python tools/verify_imagenet100_dataset.py \
  --data-root D:/datasets/imagenet100_prepared
```

This is an offline check. It validates the exact 100 WNIDs, seed-42 5x20 task artifact, all three splits, readable images, manifest identity, and split disjointness.

## Packaging and checksum

```bash
python tools/package_imagenet100_for_cluster.py \
  --data-root D:/datasets/imagenet100_prepared \
  --output D:/datasets/imagenet100_prepared.tar.gz
```

The archive contains only `train/`, `calibration/`, `test/`, and `metadata/` under `imagenet100_prepared/`; HF caches and temporary files are excluded.

Unix checksum:

```bash
sha256sum imagenet100_prepared.tar.gz
```

PowerShell checksum:

```powershell
Get-FileHash D:/datasets/imagenet100_prepared.tar.gz -Algorithm SHA256
```

Record the printed checksum in `R8/imagenet100_dataset_source_audit.md` after the real package is created. No checksum exists in this repository yet because the requested preparation stage deliberately did not download the dataset.

## Upload to the cluster

```bash
rsync -avP imagenet100_prepared.tar.gz \
  shahrampou@login:<cluster-target>/
```

or:

```bash
scp imagenet100_prepared.tar.gz \
  shahrampou@login:<cluster-target>/
```

Do not hard-code passwords.

## Cluster extraction and verification

```bash
mkdir -p /nfsd/lttm4/tesisti/shahrampour/datasets
tar -xzf imagenet100_prepared.tar.gz \
  -C /nfsd/lttm4/tesisti/shahrampour/datasets
export IMAGENET100_ROOT=/nfsd/lttm4/tesisti/shahrampour/datasets/imagenet100_prepared
python tools/verify_imagenet100_dataset.py --data-root "$IMAGENET100_ROOT"
```

The extracted dataset stays outside the Git repository. The verifier is offline and must pass before any Slurm submission.

## Benchmark submission (not executed here)

```bash
cd /nfsd/lttm4/tesisti/shahrampour/vit-cifar100-project
export IMAGENET100_ROOT=/nfsd/lttm4/tesisti/shahrampour/datasets/imagenet100_prepared
sbatch experiments_prepared/slurm/final_8method_imagenet100_5x20_job4971615_canonical_ep9.sbatch
```

The wrapper prints the host, GPU, commit, dataset counts, class/task protocol, seed, epochs, and all eight methods, then verifies the dataset before starting training. This command was not executed as part of preparation.

## Output and CPU analysis artifacts

The production run writes timestamped result tables/reports under `results/imagenet100_5x20_final_8method_canonical_job4971615_seed42_ep9_EPOCH9_MAIN_<timestamp>/` and stable resume checkpoints under the matching `_checkpoints` directory. Each method also gets `cpu_calibration_artifacts/<method>/` containing validation and test logits, labels, task IDs, sample IDs, classifier weights, and classifier biases in NumPy formats.
