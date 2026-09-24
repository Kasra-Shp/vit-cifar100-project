# ImageNet-100 dataset source audit

Status: the public source was inspected without authentication using a streaming Hugging Face load. No ImageNet archive was materialized in this code-preparation pass, and no benchmark was run.

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

## Required class-set gate

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

Because the sets differ, the downloader stops before creating or overwriting a prepared output. The public mirror is therefore not a valid source for the canonical thesis benchmark, and no public-source archive checksum exists. No alternate 100-class subset was substituted.

## Canonical transformation when an exact source is available

For an exact canonical source, the downloader preserves the WNIDs and deterministic repository class ordering, exports RGB JPEGs, and creates:

```text
imagenet100_prepared/
  train/<synset>/*.JPEG
  calibration/<synset>/*.JPEG
  test/<synset>/*.JPEG
  metadata/classes.json
  metadata/task_split.json
  metadata/dataset_manifest.json
  metadata/source.json
```

The held-out source split is deterministically divided per class using seed 42: 25 images/class for calibration and the remaining held-out images for frozen test. The manifest stores sample IDs and relative paths; calibration and test are disjoint and the verifier rejects overlap.

Use `ILSVRC/imagenet-1k` only when its ImageNet terms/access have been accepted and normal HF authentication is available, or use a licensed local ImageNet-1k directory. The downloader does not bypass gating.

## License and access

ImageNet terms and licensing restrictions remain applicable to any licensed ImageNet files. Preparation tools do not grant redistribution rights. Do not commit images, archives, caches, credentials, checkpoints, or logits.

## Checksum

The final canonical prepared archive checksum is intentionally unfilled until a compatible local/gated source is prepared:

```bash
python tools/package_imagenet100_for_cluster.py --data-root <prepared-root> --output imagenet100_prepared.tar.gz
sha256sum imagenet100_prepared.tar.gz
```

## Cluster network guarantee

The cluster production script consumes only `IMAGENET100_ROOT` and local files. It does not resolve a Hub dataset, call Hub download helpers, use HTTP clients, use `wget`/`curl`, or enable torchvision downloads. The required CLIP checkpoint must already be present in the cluster environment cache.
