# ImageNet-100 dataset source audit

Status: preparation code and documentation complete; no ImageNet archive was downloaded in this code-preparation pass.

## Source

- Provider: Hugging Face Hub, official `ILSVRC` organization.
- Exact identifier: `ILSVRC/imagenet-1k`.
- Source URL: <https://huggingface.co/datasets/ILSVRC/imagenet-1k>
- Configuration: default/main unless overridden explicitly by the downloader.
- Authentication: gated. The repository requires the user to accept ImageNet terms and authenticate normally (`huggingface-cli login` or local `HF_TOKEN`). No credentials are stored here and no access restriction is bypassed.
- Original source splits: `train` and `validation`.
- Alternative supported source: a user-provided licensed ImageNet-1k directory with `train/<synset>/` and `val/<synset>/`.

## Selected subset and transformation

The selected subset is the existing repository definition documented in `thesis_agent/reports/imagenet100_generalization_preparation.md` and verified by `scripts/verify_imagenet100_local.py`: the first 100 sorted ImageNet WNIDs, `n01440764` through `n01855672`. This is the PODNet/DER/DyTox-compatible subset, not the separate CMC/clane9 100-class subset.

The downloader selects source label IDs 0–99, retains WNID identity, converts images to RGB JPEGs, and writes deterministic filenames and a manifest. It creates:

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

The official held-out split is deterministically divided per class using seed 42: 25 images/class for calibration and the remaining held-out images for frozen test. The manifest records every sample ID and relative path, and the verifier rejects overlap.

Expected PODNet-lineage counts are approximately 129,395 selected training images and 5,000 held-out images before the 2,500/2,500 calibration/test split; the downloader records actual counts and fails if any selected class is absent.

## License and access

ImageNet terms and licensing restrictions remain applicable to the downloaded files. The preparation tools are distribution/transfer helpers, not a new license grant. Do not commit images, archives, caches, credentials, checkpoints, or logits.

## Checksum

The final prepared archive checksum is intentionally not filled in until the real local preparation occurs. Generate it with:

```bash
python tools/package_imagenet100_for_cluster.py --data-root <prepared-root> --output imagenet100_prepared.tar.gz
sha256sum imagenet100_prepared.tar.gz
```

Record the resulting SHA256 here after packaging the real data.

## Cluster network guarantee

The cluster production script consumes only `IMAGENET100_ROOT` and local metadata/files. It does not resolve a Hub dataset, call Hub download helpers, use HTTP clients, use `wget`/`curl`, or enable torchvision downloads. CLIP processor/model loading is `local_files_only=True`; the required CLIP checkpoint must already be present in the cluster environment cache.
