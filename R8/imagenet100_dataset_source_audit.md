# ImageNet-100 dataset source audit

Status: production is finalized against the existing UniPD ImageNet-1K mount. No ImageNet archive was downloaded, copied, extracted, or benchmarked in this code-preparation pass.

## Selected production source

- Provider: UniPD shared filesystem.
- Exact root: `/nfsd/lttm4/datasets/ImageNet-1k_torch`.
- Layout: `train/<WNID>/...` and `val/<WNID>/...`.
- Access: no login and no dataset network access from the cluster job.
- Selection: exactly the 100 WNIDs from `tools/imagenet100_common.py`; the other 900 ImageNet-1K classes are ignored.
- Protocol: existing deterministic seed-42 5×20 split; no runtime class resampling.
- Validation policy: 25 selected `val/` images per class for calibration and the remainder for frozen test, using seed 42. No image files are copied.

Cluster verification command:

```bash
python tools/verify_imagenet100_dataset.py \
  --data-root /nfsd/lttm4/datasets/ImageNet-1k_torch \
  --source-mode shared_imagenet1k \
  --max-readable-checks 24
```

The verifier prints `TRAIN WNIDS FOUND: 100/100`, `VAL WNIDS FOUND: 100/100`, `PROJECT SUBSET MATCH: PASS`, and `VALIDATION/TEST DISJOINT: PASS` when run on the cluster root. The exact source paths and IDs used by production are saved as `logs/shared_imagenet1k_source_manifest.jsonl`.

## Public source inspected

- Provider: Hugging Face Hub.
- Exact identifier: `clane9/imagenet-100`.
- Source URL: <https://huggingface.co/datasets/clane9/imagenet-100>.
- Public/no-auth access: PASS. Metadata and a first streaming record were accessible without `HF_TOKEN` or `huggingface-cli login`.
- Configuration: default.
- Confirmed splits: `train`, `validation`.
- Confirmed fields: `image`, `label`.
- Confirmed label representation: 100-entry `ClassLabel`; labels are integer IDs and the feature carries human-readable class names.
- Confirmed counts: train 126,689; validation 5,000.
- WNID provenance: the source label order is the CMC `imagenet100.txt` order, retained in `tools/imagenet100_common.py` as `CLANE9_IMAGENET100_SYNSETS`.
- Image representation: HF image objects; the dataset card states that mirror images were resized to a 160-pixel shorter side.

## Required class-set gate and historical public audit

The existing benchmark definition was not changed. It remains the PODNet/DER/DyTox first-100 sorted-WNID subset in `tools/imagenet100_common.py`.

```text
EXPECTED WNIDS: 100
PUBLIC DATASET WNIDS: 100
INTERSECTION: 8
MISSING FROM PUBLIC: 92 WNIDs
EXTRA IN PUBLIC: 92 WNIDs
CLASS SET MATCH: FAIL
```

The exact machine-printed lists are produced by:

```bash
python tools/download_imagenet100_external.py \
  --source hf --dataset-name clane9/imagenet-100 \
  --output-dir D:/datasets/imagenet100_prepared --seed 42 --inspect-only
```

Because the sets differ, the downloader stops before creating or overwriting a prepared output. The public mirror is therefore not a valid source for the project benchmark, and no public-source archive checksum exists. No alternate 100-class subset was substituted. The shared ImageNet-1K mount is the legitimate fallback because it contains the original WNID directories and permits exact filtering.

## Canonical transformation and production artifacts

The shared production loader preserves the WNIDs and deterministic repository class ordering without materializing a new dataset. It directly reads:

```text
ImageNet-1k_torch/
  train/<synset>/*
  val/<synset>/*
```

The held-out source split is deterministically divided per class using seed 42: 25 images/class for calibration and the remaining held-out images for frozen test. The production manifest stores exact source paths and stable sample IDs; calibration and test are disjoint and the verifier rejects overlap. Each method also saves validation/test logits, labels, task IDs, sample IDs, and classifier weights/biases for CPU-only calibration.

Use `ILSVRC/imagenet-1k` only when its ImageNet terms/access have been accepted and normal HF authentication is available, or use a licensed local ImageNet-1k directory. The downloader does not bypass gating.

## License and access

ImageNet terms and licensing restrictions remain applicable to any licensed ImageNet files. Preparation tools do not grant redistribution rights. Do not commit images, archives, caches, credentials, checkpoints, or logits.

## Checksum

No archive checksum is applicable to the preferred shared-root path. If an offline prepared copy is deliberately created, its checksum can be recorded with:

```bash
python tools/package_imagenet100_for_cluster.py --data-root <prepared-root> --output imagenet100_prepared.tar.gz
sha256sum imagenet100_prepared.tar.gz
```

## Cluster network guarantee

The cluster production script consumes only `IMAGENET100_ROOT` and local files. It does not resolve a Hub dataset, call Hub download helpers, use HTTP clients, use `wget`/`curl`, or enable torchvision downloads. The required CLIP checkpoint must already be present in the cluster environment cache. The intended audit result is:

```text
CLUSTER NETWORK DATASET ACCESS: NONE
```
