# ImageNet-100 Generalization Study — Preparation Report

**Date:** 2026-09-08 (updated same day after full local ImageNet-1k access was confirmed).
**Status: PREPARATION ONLY. NO DATA DOWNLOADED. NO JOB SUBMITTED. NO TRAINING RUN.** Sections 1-16
below are the ORIGINAL preparation pass (written when only a Hugging Face mirror was a confirmed
option); **Section 0 immediately below supersedes their dataset-choice conclusions** after a
deeper CIL-literature audit and confirmed full local ImageNet-1k access. Read Section 0 first.

---

## 0. UPDATE — Final ImageNet-100 Benchmark Selection (2026-09-08)

**Full ImageNet-1k is now confirmed available on the UniPD cluster** at
`/nfsd/lttm4/datasets/ImageNet-1k_torch` (verified `train/` and `val/` directory trees, each with
1000 WNID subdirectories). **Dataset availability is therefore no longer a selection criterion**
— this section re-derives the class-list choice from CIL-literature precedent and reproducibility
alone, per the task that requested this update, and **replaces** `IMAGENET100_SYNSETS` in
`vit_lora_cifar100_full5step_n5.py` accordingly (the original CMC list is retained in the source
as `IMAGENET100_SYNSETS_CMC_ALTERNATIVE`, inactive, for reproducibility — see Section 9 there).

### 0.1 Candidate definitions traced to primary sources (not inferred from the name)

| # | Definition | Traced to | Method |
|---|---|---|---|
| A | CMC / Tian et al. (2020) | `HobbitLong/CMC/imagenet100.txt` | Already fully recovered in the original pass (Section 2 below) |
| B | PODNet (Douillard et al., ECCV 2020) | `arthurdouillard/incremental_learning.pytorch/imagenet_split/{train,val}_100.txt` | **Fully downloaded via direct `curl`** (bypassing the prior pass's WebFetch-summarizer truncation) — both files, complete, parsed programmatically |
| C | DER (Yan et al., CVPR 2021) | `Rhyssiyan/DER-ClassIL.pytorch` | Repository's own README fetched and read directly: **"ImageNet100: Refer to [ImageNet100_Split]"**, linking straight to PODNet's repo/file above — DER defines no separate list of its own |
| D | DyTox (Douillard et al., CVPR 2022) | `arthurdouillard/dytox/imagenet100_splits/{train,val}_100.txt` | **Fully downloaded via direct `curl`**; `diff`'d byte-for-byte against PODNet's own `val_100.txt` |
| E | LUCIR (Hou et al., CVPR 2019) | `hshustc/CVPR19_Incremental_Learning` | Investigated via search only in this pass (repository's exact ImageNet-subset file was not directly fetched) — secondary sources attribute a "shuffle 1000 classes with seed 1993, take the first 100" procedure to this lineage; **could not be reconciled with the primary PODNet file's own content** (see 0.2) and is not independently confirmed here |
| F | BiC (Wu et al., CVPR 2019) / WA (Zhao et al., 2020, already `zhao2020maintaining` in this thesis's own bibliography) | Not traced to a primary repository/file in this pass | Not investigated further — out of scope once B/C/D converged on a single, well-evidenced answer |

### 0.2 Exact synset lists — extracted and verified programmatically, not prose-described

**Definition A (CMC)** — as recorded in the original Section 2/3 below: 100 WNIDs, source-file
order, verified unique/well-formed. Ordering: as-listed in the source file (author states "100
randomly selected classes," no further ordering semantics given). Class subset: fixed, sampled
once by the source authors (not re-sampled per use).

**Definitions B/C/D (PODNet/DER/DyTox)** — recovered in full by downloading
`imagenet_split/val_100.txt` (5,000 lines) and `imagenet_split/train_100.txt` (129,395 lines)
directly via `curl` (not the summarizing WebFetch tool used in the original pass, which
truncated both files before reaching label 44 of 100) and parsing every line programmatically:

- **train_100.txt: 129,395 lines** (matches the widely-cited "~129k train images" figure for this
  benchmark) — 100 distinct integer labels (0-99), each mapping to exactly one WNID.
- **val_100.txt: 5,000 lines** (50 images × 100 classes) — same 100-label structure.
- **The label→WNID mapping in `train_100.txt` and `val_100.txt` is IDENTICAL** (verified
  programmatically, all 100 entries match).
- **The resulting ordered list (`label_to_wnid[0], label_to_wnid[1], ..., label_to_wnid[99]`) is
  MONOTONICALLY INCREASING** — i.e. this is exactly `sorted(all_1000_ImageNet_WNIDs)[:100]`,
  running from `n01440764` (label 0, the alphabetically/numerically first WNID in the entire
  ImageNet-1k label set — confirmed against the standard, widely-used `imagenet_class_index`
  ordering, e.g. `n01440764`="tench", `n01443537`="goldfish", `n01484850`="great white shark", …)
  through `n01855672` (label 99).
- **This directly contradicts the "random 100-class subset, seed-1993 shuffle" description found
  in secondary/survey sources for this lineage** (Definition E above uses that exact phrase). The
  most likely reconciliation, NOT independently confirmed in this pass: the seed-1993 shuffle
  governs the INCREMENTAL TASK ORDER (which of the 100 already-selected classes appears in which
  step) in some of these papers' own experiment configs, not the SUBSET SELECTION itself, which
  for the PODNet/DyTox file lineage specifically is the plain first-100-sorted set. **This ambiguity
  is reported, not silently resolved** — the important, decision-relevant fact (the exact 100
  WNIDs actually used to build `train_100.txt`/`val_100.txt`, which is what byte-identically
  reproduces PODNet/DER/DyTox's own experiments) is fully verified regardless of which narrative
  about its origin is correct.
- **DyTox's own copy of `val_100.txt` is byte-for-byte identical** (`diff` = 0 lines) to PODNet's.
  **DER's README explicitly defers to PODNet's file rather than defining its own.** ⇒ **PODNet,
  DER, and DyTox provably use the exact same 100-class subset and the exact same label indexing.**

The complete, verified 100-WNID list (ascending order, as extracted) is stored in
`vit_lora_cifar100_full5step_n5.py` as `IMAGENET100_SYNSETS` (Section 0.6 shows the loader-time
policy for turning this into training labels).

### 0.3 CMC vs. chosen-CIL-subset — literal programmatic comparison

```
CMC count: 100 (100 unique)      PODNet/DER/DyTox count: 100 (100 unique)
Shared classes:        8
Only in CMC:          92
Only in PODNet/DER/DyTox: 92
Union:                192
Jaccard overlap = 8/192 = 0.0417
Exactly identical (as sets):    False
Exactly identical (as lists):   False
```

Shared WNIDs (8): `n01558993, n01692333, n01729322, n01735189, n01749939, n01773797, n01820546,
n01855672`. **A Jaccard overlap of 0.0417 is close to the ~0.0526 expected by pure chance for two
independent random 100-of-1000 subsets** — these are, for practical purposes, **two unrelated
benchmarks that happen to share a name**, not two orderings or minor variants of the same one.

### 0.4 Scientific selection — PODNet/DER/DyTox chosen

Against the task's stated priority order:

1. **CIL literature precedent:** PODNet/DER/DyTox lineage wins decisively — three separate,
   highly-cited class-incremental-learning papers (ECCV'20, CVPR'21, CVPR'22) confirmed (not
   assumed) to share one subset. CMC is a self-supervised/contrastive-representation-learning
   benchmark with no comparable CIL pedigree.
2. **Reproducibility:** now EQUAL for both (both are fully, programmatically recovered in this
   pass) — this axis no longer favors CMC as it did in the original preparation pass (which could
   not fully extract the PODNet-lineage list at the time).
3. **Exact published/repository-backed definition:** both qualify; PODNet/DyTox's is
   cross-confirmed byte-identical across two independent repositories, a slightly stronger
   provenance chain than CMC's single-repository source.
4. **Compatibility with 100 classes / 4×25:** both are exactly 100 classes; no difference.
5. **Compatibility with full local ImageNet-1k:** **decisive tie-breaker removed as a
   discriminator per this task's own instruction** — both are now equally satisfiable, since all
   200 or so distinct WNIDs across both lists are ordinary ImageNet-1k classes expected to exist
   in any standard `ILSVRC2012` install (verification in Section 0.5).

**Final recommendation: PODNet/DER/DyTox lineage (`IMAGENET100_SYNSETS`, Section 0.2).** This
reverses the original pass's CMC-by-default choice, which was explicitly conditioned on "IF the
author confirms ImageNet-1k *is* available on the training cluster, switching to [this] lineage...
is a live alternative worth revisiting" — that condition is now met.

### 0.5 Local cluster dataset verification

A standalone, read-only verification script was written:
`scripts/verify_imagenet100_local.py`. For each of the 100 chosen WNIDs it checks
`<IMAGENET_ROOT>/train/<wnid>/` and `<IMAGENET_ROOT>/val/<wnid>/` exist and are non-empty, and
reports any missing/empty classes. It does **not** read, copy, or move any image, and does not
touch any directory outside the 100 selected WNIDs or any "snapshot" subdirectory.

**This script's logic was sandbox-tested** (a temporary mock directory tree with all 100 WNIDs,
one deliberately removed from `val/`) and correctly identified exactly the one missing class and
nothing else. **It has NOT been run against the real cluster path** — this session has no access
to `/nfsd/lttm4/datasets/ImageNet-1k_torch`. Running it there (`python
scripts/verify_imagenet100_local.py`, optionally with `IMAGENET_ROOT=...`) is the next concrete
step, and is expected to report 0 missing classes given all 100 chosen WNIDs are ordinary
ImageNet-1k classes, but this expectation is not yet a verified fact.

### 0.6 Final class order (4×25 protocol)

No canonical *per-step task order* was confirmed to accompany this benchmark in this
investigation (PyCIL/PODNet-family configs reference a "class order" concept in their own
tooling, but the exact published values were not extracted in this session — see Section 0.2's
own caveat about the seed-1993 description). Per the task's explicit fallback instruction, this
plan uses the project's own canonical `SEED=42` to generate **one** deterministic permutation of
the 100 chosen WNIDs (Python's `random.Random(42).shuffle()` applied to the WNID-sorted
`IMAGENET100_SYNSETS` list), stored in full as `IMAGENET100_CLASS_ORDER` in the source file. This
is not a semantically convenient or cherry-picked grouping — every position was assigned purely by
that one deterministic shuffle.

Resulting 4×25 groups (by position in `IMAGENET100_CLASS_ORDER`, recorded here and in the source):

- **Step 1 (positions 0-24):** n01687978, n01685808, n01824575, n01518878, n01751748, n01698640,
  n01443537, n01770081, n01558993, n01776313, n01773549, n01530575, n01729977, n01734418,
  n01773157, n01692333, n01695060, n01828970, n01774750, n01669191, n01641577, n01608432,
  n01644900, n01843383, n01795545
- **Step 2 (positions 25-49):** n01697457, n01798484, n01630670, n01817953, n01664065, n01514859,
  n01694178, n01739381, n01748264, n01773797, n01689811, n01855032, n01728572, n01806567,
  n01532829, n01667778, n01616318, n01677366, n01682714, n01582220, n01753488, n01742172,
  n01740131, n01514668, n01665541
- **Step 3 (positions 50-74):** n01855672, n01693334, n01484850, n01704323, n01560419, n01675722,
  n01737021, n01756291, n01614925, n01744401, n01622779, n01496331, n01498041, n01755581,
  n01797886, n01592084, n01784675, n01688243, n01820546, n01601694, n01440764, n01843065,
  n01735189, n01829413, n01728920
- **Step 4 (positions 75-99):** n01819313, n01629819, n01770393, n01806143, n01775062, n01749939,
  n01632777, n01631663, n01818515, n01847000, n01494475, n01729322, n01774384, n01531178,
  n01768244, n01807496, n01534433, n01580077, n01632458, n01644373, n01667114, n01833805,
  n01491361, n01537544, n01796340

### 0.7 Loader strategy (revised — local, not HF Hub)

`vit_lora_cifar100_full5step_n5.py` was revised to add `DATASET_REGISTRY["imagenet100"]["loader"]
= "local_imagefolder"` (replacing the earlier `"hf_hub"`/`clane9/imagenet-100` plan, which is now
unused and superseded — no HuggingFace download remains anywhere in the imagenet100 launch path):

- `IMAGENET_ROOT` is read from the environment (`os.environ.get("IMAGENET_ROOT",
  "/nfsd/lttm4/datasets/ImageNet-1k_torch")`) — configurable, not hardcoded to any user's home
  directory, defaulting to the verified cluster path.
- `load_dataset("imagefolder", data_dir=IMAGENET_ROOT, data_files={"train": [f"train/{wnid}/*" for
  wnid in IMAGENET100_SYNSETS], "validation": [f"val/{wnid}/*" for wnid in IMAGENET100_SYNSETS]})`
  — filters the shared ImageFolder tree to exactly the 100 selected WNIDs via glob patterns; the
  other 900 ImageNet-1k classes are never opened, read, or copied. Nothing is duplicated on disk.
- **Critical label remap (the exact risk the task flagged):** HF's `imagefolder` builder assigns
  its own `ClassLabel` indices (0-99) based on the alphabetically-sorted set of the 100 discovered
  class subfolders — this is NEVER assumed to already equal `IMAGENET100_CLASS_ORDER`. The code
  explicitly reads back `dataset["train"].features[LABEL_COL].names` (recovering each loader-index
  → WNID mapping), builds `IMAGENET100_WNID_TO_LABEL` from `IMAGENET100_CLASS_ORDER`'s own position
  order, and applies `dataset.map(_remap_imagenet100_label)` to overwrite every example's label
  with the WNID's position in `IMAGENET100_CLASS_ORDER` — guaranteeing the classifier never
  silently trains against the loader's own indices or the original 0-999 ImageNet-1k indices.
- A loader-time assertion verifies the discovered class set (100 classes, matching
  `IMAGENET100_SYNSETS`) before any remap is applied, and fails loudly (not silently) on a
  mismatch.
- **This remap logic is implemented and statically reviewed but UNTESTED against real data** (no
  cluster access from this session) — flagged explicitly in the source code's own comments and in
  Section 12/Risks below as the first thing to check via `Counter(dataset["train"][LABEL_COL])`
  once a real load is possible (expect exactly 100 distinct values, 0..99, non-degenerate
  per-class counts).
- Everything downstream (`filter_by_classes`, `build_classwise_train_val_splits`,
  `preprocess_train`/`preprocess_val`, `class_splits`) is completely unchanged and operates on the
  now-correctly-remapped integer labels exactly as it already does for cifar100.

---

## 1. Why ImageNet-100

The canonical 8-method CIFAR-100 comparison (R6-15, replicated at seed123 as R6-17) establishes
several conclusions on a single visual dataset. The scientific purpose of this study is to test
whether those conclusions are dataset-specific or generalize: does the large SimpleAvg-vs-RankExt
gap persist, does KD remain the dominant RankExt stabilizer, does FactorOrth remain a smaller
effect, on a *substantially different* natural-image dataset?

ImageNet-100 was chosen over the alternatives this project had previously assessed (see
`reports/next_dataset_backbone_assessment.md`, which recommended **ImageNet-R** for a *domain-shift*
generalization test) because it uniquely offers **100 classes**, letting the entire 4×25 protocol,
RankExt's cumulative-rank trajectory ([20,40,60,80]), and the total-class-count assumption
(`NUM_CLASSES=100`) carry over completely unchanged. ImageNet-R (200 classes) or Tiny-ImageNet
(200 classes) would require a different protocol shape, confounding "new dataset" with "new
protocol." This makes ImageNet-100 the more surgical choice for isolating *dataset identity* as
the single changed variable — a different, complementary generalization axis to ImageNet-R's
domain-shift axis, not a replacement for it.

---

## 2. Exact benchmark/subset definition — INVESTIGATED, NOT SILENTLY CHOSEN

**The project's own prior assessment already flagged the exact risk this section addresses:**
`reports/next_dataset_backbone_assessment.md` (written before this task) states ImageNet-100 is
"**Gated** — needs ImageNet-1k license/access agreement; not a clean single HF id, **several
incompatible 'ImageNet-100' class lists circulate in the literature**" and recommended against it
in favor of ImageNet-R for that report's own (different) research goal. This investigation
confirms that finding empirically and resolves it for *this* task's specific goal (protocol
preservation), rather than re-litigating that earlier, differently-scoped recommendation.

**At least three distinct, real, named "ImageNet-100" definitions were found:**

| Definition | Origin | Class selection | Verified how | Availability |
|---|---|---|---|---|
| **A. CMC / Tian et al. (2020)** | "Contrastive Multiview Coding," `HobbitLong/CMC` GitHub repo, `imagenet100.txt` | 100 classes described by the source as "randomly selected" from ImageNet-1k | **Fetched the raw file directly** (`raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt`); programmatically verified as exactly 100 entries, all unique, all well-formed WNIDs (`n\d{8}`) | Mirrored, **ungated**, directly `load_dataset()`-able on Hugging Face Hub as `clane9/imagenet-100` (verified: HF API returns `"gated":false,"private":false`) and as Kaggle datasets (`ambityga/imagenet100`, `sidharthangn/imagenet-100`) |
| **B. PODNet / DER / DyTox lineage** | `arthurdouillard/incremental_learning.pytorch`, `imagenet_split/{train,val}_100.txt`; reused by DER (Yan et al., CVPR 2021), DyTox (Douillard et al., CVPR 2022), and multiple later CIL papers | A per-image manifest (label 0..99); the first ~44 labels inspected resolve to WNIDs in ascending numeric order starting at `n01440764` (ImageNet's first synset) — i.e. this looks like "classes 0-99 in default WNID sort order," not a random subset, though the full 100-entry set was **not** completely extracted (see Limitations) | **Partially verified** — file located and its *structure* and *first 44 of 100 classes* directly fetched and inspected; the complete distinct 100-class list was not fully extracted due to the manifest being a large (~130k-line) per-image file rather than a compact per-class list | Requires a full local/cluster ImageNet-1k (ILSVRC-2012) install to reconstruct (the manifest is paths into a full ImageNet-1k directory tree) — **no standalone downloadable mirror found** |
| **C. iCaRL / UCIR "seed-1993" lineage** | Rebuffi et al. 2017 (iCaRL) originated the CIL "ImageNet subset" protocol; Hou et al. 2019 (UCIR) and others fixed a random-seed-1993 class-order shuffle convention, also present as `train_100_ucir.txt` in the PODNet repo above | 100 classes selected via a fixed-seed (1993) random permutation of ImageNet-1k's class order | **Not independently fetched/verified in this pass** (found only via secondary description in search results, not a raw source file) | Same access requirement as B (full ImageNet-1k) |

**These three are confirmed NOT interchangeable** (definition A's first listed WNID is
`n02869837`; definition B's label-0 WNID is `n01440764` — different classes at the same index).
**No subset was silently chosen without this comparison being surfaced.**

### Recommendation

**Primary, currently wired into the code (`DATASET_REGISTRY["imagenet100"]`, `IMAGENET100_SYNSETS`
in `vit_lora_cifar100_full5step_n5.py`): Definition A (CMC / Tian et al. 2020).**

Reasoning against the four requested criteria:

- **Prevalence in CIL literature:** LOWER than Definition B specifically (B is the modern
  class-incremental-learning-specific standard, reused by DER/DyTox/PODNet and their many
  follow-ups). This is a real point against A.
- **Reproducibility:** HIGH for A (a complete, verified, static 100-WNID list from a stable,
  citable source) vs. MEDIUM for B (the exact list was not fully extracted in this pass — see
  Limitations — and reconstructing it requires deriving the distinct-class set from a ~130k-line
  manifest rather than reading a clean list).
- **Compatibility with this project:** the existing pipeline calls `load_dataset("cifar100")`
  once and lets HF's `datasets` library handle everything else. Definition A is available as a
  single ungated `load_dataset("clane9/imagenet-100")` call — an almost drop-in match for the
  existing code path. Definition B requires a full ImageNet-1k install plus custom
  `ImageFolder`/manifest-based loading code not currently present anywhere in this project.
- **Availability of exact class IDs:** COMPLETE for A (recorded in full, see `IMAGENET100_SYNSETS`
  in the source and Section 12's verification caveat); INCOMPLETE for B in this pass.

**This recommendation is explicitly conditional on Part E's finding (Section 4): full ImageNet-1k
access from the actual training cluster was not verified in this pass.** If the author confirms
ImageNet-1k *is* available on the training cluster, switching to Definition B (the
literature-prevalence-preferred choice for a CIL thesis specifically) is a live alternative worth
revisiting before launch — it would require additional implementation work (a custom
manifest-driven loader) not done in this pass. **This report does not decide that trade-off for
the author; it is flagged here for explicit review**, per the task's own instruction not to
silently choose among incompatible definitions.

---

## 3. Source of class list

- **Chosen (Definition A):** `https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt`
  (Tian, Y., Krishnan, D., Isola, P. "Contrastive Multiview Coding," 2020). Mirrored, with the
  identical class set, as `clane9/imagenet-100` on the Hugging Face Hub (`load_dataset()`-compatible)
  and as `ambityga/imagenet100` / `sidharthangn/imagenet-100` on Kaggle.
- **Alternative (Definition B):** `https://github.com/arthurdouillard/incremental_learning.pytorch`,
  `imagenet_split/train_100.txt` and `val_100.txt`.
- **Alternative (Definition C):** iCaRL (Rebuffi et al., CVPR 2017) protocol lineage; UCIR
  (Hou et al., CVPR 2019) `train_100_ucir.txt` variant in the same PODNet repo.

---

## 4. Dataset availability — AUDITED, NOT ASSUMED

| Check | Result |
|---|---|
| Existing ImageNet path/config anywhere in this repo | **NOT FOUND** — grepped the entire project tree (`.py`/`.json`/`.yaml`/`.sh`) for "imagenet"; no hits outside this session's own new edits |
| SLURM/cluster config files in this repo | **NOT FOUND** — no `.slurm`/`.sbatch`/cluster env files present locally; this desktop environment is evidently only where the source script is edited and results are analyzed, not where training executes (confirmed separately: local `torch.cuda.is_available()` is `False`, this machine has no GPU) |
| ImageNet-1K training data locally | **NO** |
| ImageNet validation data locally | **NO** |
| An existing ImageNet-100 subset locally | **NO** |
| Network reachability to the chosen HF mirror, from this local machine | **YES** — `GET https://huggingface.co/api/datasets/clane9/imagenet-100` returned HTTP 200 with `"gated":false,"private":false,"disabled":false` |
| Required Python packages present locally | **YES** — `datasets` 4.6.1, `torch` 2.10.0 (CPU-only), `transformers` 5.1.0 all import successfully |
| ImageNet-100 (or any ImageNet) data actually downloaded in this pass | **NO — deliberately not attempted**, per explicit instruction not to download ImageNet automatically and to let the author review the class-list choice first |

**Conclusion: ImageNet data are NOT currently accessible in this environment; the target HF
mirror is confirmed reachable in principle from this machine, but this says nothing about the
actual training cluster's network policy.**

**What must be checked on the cluster before launch:**
1. Outbound internet access to `huggingface.co` (and its CDN, `cdn-lfs.huggingface.co` /
   `*.hf.co`) from the compute nodes that will actually run the job — many HPC clusters block or
   heavily restrict compute-node internet access, requiring either a login-node pre-download step
   or an offline dataset cache staged in advance.
2. Available local/scratch disk quota for the dataset cache (`~/.cache/huggingface/datasets` by
   default) — estimated 3-8 GB for this specific mirror (Section 10), not independently confirmed
   against the HF Hub's own reported dataset size in this pass.
3. If Definition B is chosen instead (Section 2), full local ILSVRC-2012 (ImageNet-1k) train+val
   access under whatever path convention the cluster uses — this project has no existing
   convention for that path, so one would need to be established from scratch.

---

## 5. Train/val/test design

Mirrors the existing CIFAR-100 pipeline's own policy exactly, with one necessary renaming fix:

- **CIFAR-100 (existing, unchanged):** HF splits are `{"train": 50000, "test": 10000}`.
  `dataset["test"]` is the final, never-used-for-selection evaluation set. A model-selection
  validation set is carved out of `dataset["train"]` by `build_classwise_train_val_splits()`
  (25 images/class, deterministic via `SEED + int(cls)` per-class shuffling, class-balanced by
  construction) — this function is fully dataset-agnostic already and needed **no changes**.
- **ImageNet-100 (`clane9/imagenet-100`, planned):** HF splits are `{"train": 126689,
  "validation": 5000}` — **there is no split literally named "test."** Per the task's own
  instruction ("do NOT create an arbitrary test split if the official validation set can serve as
  evaluation... final ImageNet validation images must not be used for hyperparameter/model-selection
  decisions if they are the final test/evaluation set"), this dataset's `"validation"` split (=
  ImageNet's own official validation set) is treated as the **final evaluation split** — analogous
  to CIFAR's `"test"` — and is never touched by model selection. A `DATASET_REGISTRY["final_eval_split"]`
  indirection (`"test"` for cifar100, `"validation"` for imagenet100) was added so
  `make_eval_dataset()` reads the correct split per dataset without duplicating logic. The
  model-selection validation set is still carved out of `dataset["train"]` by the exact same
  `build_classwise_train_val_splits()` call, unchanged, with `VALIDATION_PER_CLASS=25` left at its
  existing value (not retuned) — 25/class is well within the smallest class's ~1200+ average
  training-image count for this mirror.

---

## 6. Preprocessing

**Audited the existing CIFAR-specific transform pipeline — it required NO changes.** The pipeline
already derives its target resolution and normalization constants from the CLIP checkpoint itself
(`CLIPImageProcessor.from_pretrained(MODEL_CHECKPOINT)` → `H=W=224`, `image_mean`/`image_std` from
CLIP, not hardcoded), and `transforms.Resize((H, W))` is the first step of both the train and eval
transform chains — meaning it already upsamples CIFAR's native 32×32 to 224×224, and will
identically resize ImageNet-100's (pre-resized-to-160px-shorter-side) images to 224×224. There is
**no 32×32-specific assumption anywhere in this pipeline.**

- **Training augmentation** (`train_transform`): `Resize→RandomCrop(padding=8)→RandomHorizontalFlip→
  ColorJitter(0.05,0.05,0.05)→ToTensor→Normalize`. `RandomCrop(..., padding=8)` is a
  small-image-community-style augmentation (padding=8 of 224 is a modest ~3.5% margin) that has
  not been separately re-validated for natural ImageNet-style photos. Per the explicit instruction
  **not to retune hyperparameters before the first portability run**, this is left unchanged for
  the first transfer test — flagged here as a known carry-over, not silently altered.
- **Evaluation preprocessing** (`val_transform`, used for both the model-selection validation set
  and the final eval set): `Resize→ToTensor→Normalize` — no augmentation, correctly separated from
  the training path already.
- **Image decoding** (`to_pil()`): already handles PIL images, HF's dict-with-`bytes`/`array`
  encodings, and raw numpy arrays defensively — this covers the JPEG-bytes-in-parquet format the
  `clane9/imagenet-100` mirror uses without modification.

---

## 7. Code portability changes (what was actually audited and what changed)

Audited every item the task listed:

| Item | Finding | Change needed |
|---|---|---|
| `torchvision.datasets.CIFAR100` | **Not used** — the pipeline uses `datasets.load_dataset`, not `torchvision`, so there is no torchvision-specific CIFAR class to generalize | None |
| Class count = 100 | Already a named constant (`NUM_CLASSES=100`), shared by both datasets (100 is exact for both) | None (added an assertion, Section 12) |
| Class names | **Never used anywhere** in the script (grepped for `fine_label_names`/`class_names`/`ClassLabel` usage in plotting/reporting code — zero hits); all reporting uses numeric class IDs or method names | None |
| 32×32 assumptions | **None found** — see Section 6 | None |
| Data paths | Loaded via `load_dataset(<hf_id>)`, no filesystem path hardcoded for either dataset | Changed the HF id from a literal `"cifar100"` to `DATASET_REGISTRY["hf_id"]` |
| Transforms | Already resolution/dataset-agnostic (Section 6) | None |
| Classifier dimensions | `CLIPVisionForCIFAR100.__init__(self, checkpoint, num_labels)` — `num_labels` is a parameter, `nn.Linear(hidden_size, num_labels)` has no hardcoded 100. The class's *name* says "CIFAR100" but this is cosmetic only | None functional (name left as-is — renaming a class used throughout a 9000-line file is exactly the kind of unrelated-infrastructure churn the task warned against) |
| Step grouping (`class_splits`) | Already derived generically from `NUM_STEPS`/`CLASSES_PER_STEP`, chunks native label order 0..99 contiguously, no shuffling — this policy is preserved for imagenet100 (Section 8) | None |
| Plotting labels | Use method names / step indices, not dataset-specific strings, in every plot function inspected | None |
| Report names / result directory naming | `RUN_NAME_BASE` previously hardcoded `"cifar100"` and the seed123 replication's literal wording | **Changed** — see Part A/B of this task; now `f"clip_vit_lora_{DATASET_NAME}_4x25_{EXPERIMENT_LABEL}_seed{SEED}"` |
| Cached metadata | `dataset["train"].column_names` / `.features` read defensively already (`LABEL_COL`/`IMAGE_COL` fallback logic predates this task) | None |
| Class-order logic | Contiguous, unshuffled chunking of native label order — see Section 8 for what "native label order" means for imagenet100 specifically | None (policy documented, not code) |
| Validation construction | `build_classwise_train_val_splits()` fully generic | None (see Section 5 for the one related change: `final_eval_split` split-name indirection) |

**Net functional code change:** one dataset-id dispatch (`load_dataset(DATASET_REGISTRY["hf_id"])`),
one final-eval-split-name indirection (`dataset[DATASET_REGISTRY["final_eval_split"]]`), the new
`DATASET_REGISTRY`/`IMAGENET100_SYNSETS` constants plus their own internal consistency assertions,
and one consolidated experiment-invariant assertion block. **SimpleAvg, RankExt, KD, FactorOrth,
calibration, warmup/anchor/protection logic, and every evaluation-metric computation were not
touched.**

---

## 8. Class ordering (Part F)

No published class *ordering* (as opposed to class *membership*) accompanies the CMC/Definition-A
list beyond its own as-listed file order (which the HF mirror's dataset card states is **not**
what determines the mirror's internal label indices — those follow the *sorted* WNID list
instead). Per the task's stated preference ("a published ordering if one accompanies the chosen
benchmark; otherwise a documented deterministic permutation"), and because no such published
per-step ordering exists for this benchmark in a CIL context, this plan uses the **same policy the
existing CIFAR-100 code already uses and does not shuffle**: classes are chunked contiguously by
whatever integer label index the loaded HF dataset assigns (0-24 → step 1, 25-49 → step 2, 50-74 →
step 3, 75-99 → step 4), with **zero cherry-picking** — this is mechanically identical to
`class_splits`'s existing, unmodified implementation. The loader-time assertion added in Section 7
verifies the *loaded* class set matches `IMAGENET100_SYNSETS` (membership), and the exact
label-index → WNID mapping actually assigned by the HF dataset builder will be printed and
recorded the first time the dataset is actually loaded (this happens automatically via the
existing `print("Dataset columns:", ...)` / class-splits print block once the run is actually
started) — deferred to that point rather than guessed here, since it depends on the HF dataset
builder's own indexing behavior, not on anything this project controls.

---

## 9. Method-set options (Part L) — evaluated, not decided by convenience

| | Option A (8 methods) | Option B (5 methods) |
|---|---|---|
| Methods | SimpleAvg ×4 + RankExt ×4 | SimpleAvg (plain only) + RankExt ×4 |
| Answers RQ1 (SimpleAvg vs. RankExt)? | Yes | Yes |
| Answers RQ2 (RankExt KD×FactorOrth ablation)? | Yes (complete 2×2) | Yes (complete 2×2 — identical RankExt-side methods in both options) |
| Tests whether SimpleAvg-side stabilization (KD hurts, FactorOrth neutral — Section 8 of the
  seed-replication report) generalizes? | **Yes** | No — only plain SimpleAvg is run |
| Relative compute (Section 10) | ~1.6× Option B (adds 3 more independently-trained-per-step SimpleAvg variants) | baseline of this comparison |

**Recommendation: Option A (8 methods), if the compute estimate (Section 10) is acceptable to the
author.** The seed42/seed123 replication just completed found the SimpleAvg-side KD-hurts/FactorOrth-neutral
pattern to be one of the more surprising, non-obvious findings in the whole comparison (Section 8
of `r6_seed42_seed123_replication_analysis.md`) — precisely the kind of result worth checking for
dataset-specificity, not just the RQ1/RQ2 headline patterns. If compute or time budget is tight,
**Option B is fully sufficient for the primary RQ1/RQ2 scientific purpose** stated in Section 1 and
is the defensible fallback.

---

## 10. Compute estimate (Part P)

Baseline: R6-15/R6-17 (CIFAR-100, 8 methods, 7 epochs) ≈ **1.0×**.

**Scaling factors identified (not fabricated GPU-hours):**
- **Data volume:** ImageNet-100 (`clane9/imagenet-100`) has 126,689 train images vs. CIFAR-100's
  50,000 — a **~2.53×** increase in images iterated per epoch (both datasets use the same
  contiguous 4×25 class chunking and the same `VALIDATION_PER_CLASS=25` absolute carve-out, so this
  ratio applies roughly uniformly per step).
- **Per-image GPU compute:** **unchanged** — both datasets are resized to the same 224×224 CLIP
  input; forward/backward FLOPs per image are identical regardless of source dataset.
- **Data-loading/decode overhead:** ImageNet-100 images are JPEG-encoded (even at this mirror's
  reduced 160px-shorter-side resolution) vs. CIFAR-100's already-decoded small arrays — real but
  not precisely quantifiable without an actual dataloader benchmark on the target hardware; assumed
  a **1.2-1.5×** additional per-image overhead multiplier on top of the volume increase, based on
  typical JPEG-decode-vs-raw-array cost ratios, not measured here.
- **Final-eval set:** ImageNet-100's final eval split (5,000 images) is *smaller* than CIFAR-100's
  (10,000) — a small offsetting reduction.

**Combining these (data volume × decode-overhead range), the estimated per-epoch wall-time
multiplier is approximately 2.5×-4×** relative to the CIFAR-100 baseline, for either method-set
option (the multiplier applies per-method, so it does not change between Option A/B; only the
*number of methods run* does).

| | Estimated relative runtime vs. R6-15 (1.0×) |
|---|---|
| ImageNet-100, 5 methods | **~1.6× × 2.5-4× ≈ 4-6.4×** of R6-15's own total wall-clock (5/8 of the methods, at 2.5-4× per-method cost) |
| ImageNet-100, 8 methods | **~2.5-4×** of R6-15's own total wall-clock (all 8 methods, at 2.5-4× per-method cost) |

**This is a range, not a fabricated precise GPU-hour figure**, because (a) the actual target GPU's
throughput was never recorded in any inspected config/report in this project (per
`canonical_3seed_runtime_plan.md`'s own prior finding), and (b) the JPEG-decode overhead multiplier
is a reasonable estimate, not a measurement. **The most reliable way to narrow this range is a
short, real dataloader-throughput micro-benchmark once the dataset is actually downloaded** — this
was explicitly not attempted in this pass (no data was downloaded).

---

## 11. Storage estimate (Part P)

- **New one-time cost: the ImageNet-100 dataset cache itself.** `clane9/imagenet-100` is
  documented as pre-resized to 160px-shorter-side JPEGs; based on the image count (131,689 total)
  and typical compression at that resolution, a reasonable estimate is **3-8 GB** on disk
  (`~/.cache/huggingface/datasets` by default) — not confirmed against the HF Hub's own reported
  repo size in this pass (no download was attempted).
- **Per-run "light" result-package storage** (tables/plots/configs/logs — the kind of artifact
  actually synced back to this local machine for the existing CIFAR-100 runs): measured directly
  from the just-completed seed123 run (`R6/canonical_seed123_results/.../`) at **108 MB total**
  for a full 8-method run (`models/`=24 KB, `tables/`=624 KB, `plots/`=6.8 MB — the bulk is
  elsewhere, e.g. `logs/training_loss_history_by_batch.csv`). This footprint is **dataset-size-independent**
  (row/plot counts scale with methods×steps×epochs, not with the source image dataset), so the
  ImageNet-100 run's own light-result package should be a similar order of magnitude, ~100-150 MB.
- **Full checkpoint storage** (LoRA adapter weights, not included in the "light" packages synced
  locally): unknown from local evidence — this project's local copies never include full
  checkpoints for any run inspected. LoRA adapter size depends on rank/target-modules/architecture,
  **not** on the source image dataset, so it should be similar in order of magnitude to whatever
  the CIFAR-100 canonical run's own (cluster-side, not locally visible) checkpoint footprint is —
  ask the cluster for that reference figure rather than treating this report's estimate as
  authoritative.

---

## 12. Smoke-test result (Part T)

**Data was deliberately NOT downloaded in this pass** (per explicit instruction and the task's own
final "stop for review" request), so the dataset-instantiation / batch-inspection / forward-pass
portions of the allowed smoke test are **NOT POSSIBLE** yet and are marked **PENDING** actual data
access (either local, after author approval, or on the cluster).

**Source/static checks that WERE performed and passed:**

| Check | Result |
|---|---|
| `python -m py_compile vit_lora_cifar100_full5step_n5.py` (full-file syntax check) | **PASS** |
| Isolated dry-run of the new config/assertion logic (`DATASET_REGISTRY`, `IMAGENET100_SYNSETS`, the consolidated experiment-invariant assertions) for both `DATASET_NAME="cifar100"` and `"imagenet100"`, without executing the real heavy script | **PASS** for both — resolves to `RUN_NAME_BASE='clip_vit_lora_cifar100_4x25_canonical_seed42'` (cifar100 default) and `RUN_NAME_BASE='clip_vit_lora_imagenet100_4x25_generalization_seed42'` (imagenet100, matching the task's own example exactly) |
| `IMAGENET100_SYNSETS` internal consistency (100 entries, all unique, all well-formed WNIDs) | **PASS** (verified twice: once at fetch time, once in the isolated dry-run) |
| `USE_RANKEXT_RANK_SCHEDULE_WIDE == False` and `active_rankext_rank_schedule() == [20,40,60,80]` | **PASS** |
| `MODEL_CHECKPOINT == "openai/clip-vit-base-patch16"` (backbone unchanged) | **PASS** |
| Network reachability to the chosen HF mirror from this local machine | **PASS** (HTTP 200, ungated) |
| Required Python packages importable locally (`datasets`, `torch`, `transformers`) | **PASS** |
| Actual `load_dataset("clane9/imagenet-100")` call | **NOT ATTEMPTED** (would download real data) |
| Batch/tensor-shape inspection, LoRA q_proj/v_proj injection check, one forward pass | **NOT POSSIBLE without the above** |

**Overall smoke-test classification: PARTIAL — every check possible without touching real
ImageNet data passed; the data-dependent checks are pending author approval to fetch the dataset.**

---

## 13. Risks

1. **Class-list choice (Section 2) is the single highest-leverage open decision** — proceeding with
   Definition A (CMC) trades literature-prevalence for reproducibility/access-ease; this must be a
   conscious author choice, not a default that goes unnoticed.
2. **PODNet/DER-lineage list (Definition B) was not fully extracted** in this pass — if the author
   prefers it, additional work (full-manifest fetch/parse, or cluster-side extraction from a real
   ImageNet-1k install) is needed before it can be wired in the same way Definition A now is.
3. **Cluster network/access policy is unverified** — Section 4's cluster-side checklist must be
   confirmed before any actual launch; if compute nodes cannot reach the HF Hub, the dataset must
   be pre-staged another way.
4. **Compute/storage estimates are ranges derived from documented dataset sizes and general
   JPEG-decode-cost reasoning, not a measured benchmark** — treat Section 10/11 as planning inputs,
   re-estimate after a real dataloader throughput check once data is available.
5. **`RandomCrop(padding=8)` and the other training-time augmentation constants were carried over
   unchanged** (per explicit instruction not to retune before the first transfer test) — if the
   first ImageNet-100 run's results look anomalous, this is one of the first places to
   reconsider, but it should not be pre-emptively changed.
6. **The CMC mirror's images are pre-resized to 160px shorter side**, not full native ImageNet
   resolution — this is a real, if minor, departure from "native ImageNet-scale images" and should
   be stated plainly in any eventual write-up rather than implied to be full-resolution ImageNet.
7. **ImageNet's license is non-commercial research/educational use** — compatible with thesis use,
   but should be cited accurately (not silently omitted) wherever this dataset is eventually
   discussed in the thesis.

---

## 14. Exact launch configuration (prepared, NOT executed) — UPDATED per Section 0

**Supersedes this section's original (HF Hub / CMC) version below the line** now that the local
ImageNet-1k path has been confirmed and the class list finalized to the PODNet/DER/DyTox lineage.

```
DATASET_NAME       = "imagenet100"
EXPERIMENT_LABEL    = "generalization"
SEED                = 42                          # external REPLICATION_SEED override still available
RUN_NAME_BASE       = "clip_vit_lora_imagenet100_4x25_generalization_seed42"
MODEL_CHECKPOINT    = "openai/clip-vit-base-patch16"   # unchanged
NUM_CLASSES         = 100
NUM_STEPS           = 4
CLASSES_PER_STEP    = 25
RANKEXT_RANK_SCHEDULE            = [20, 40, 60, 80]     # unchanged, canonical
USE_RANKEXT_RANK_SCHEDULE_WIDE   = False                # must stay False
DATASET_REGISTRY["imagenet100"]["loader"]           = "local_imagefolder"   # was "hf_hub"
DATASET_REGISTRY["imagenet100"]["final_eval_split"] = "validation"
IMAGENET_ROOT        = "/nfsd/lttm4/datasets/ImageNet-1k_torch"  # env-var IMAGENET_ROOT overridable
IMAGENET100_SYNSETS  = <100 WNIDs, PODNet/DER/DyTox lineage, Section 0.2 -- sorted(all_1000)[:100]>
IMAGENET100_CLASS_ORDER = <same 100 WNIDs, seed-42 permutation, Section 0.6>
VALIDATION_PER_CLASS = 25                          # unchanged, not retuned
Method set           = OPTION A (8 methods) recommended, OPTION B (5 methods) acceptable fallback
```

To actually launch (not done): set `DATASET_NAME = "imagenet100"` and `EXPERIMENT_LABEL =
"generalization"` at the top of `vit_lora_cifar100_full5step_n5.py` (currently `"cifar100"` /
`"canonical"`), run `scripts/verify_imagenet100_local.py` on the cluster and confirm 0 missing
classes (Section 0.5), run the `Counter(dataset["train"][LABEL_COL])` sanity check noted in
Section 0.7, then submit exactly as the existing CIFAR-100 jobs are submitted. **No HuggingFace
download is part of this path.**

<details>
<summary>Original (superseded) launch configuration, preserved for reproducibility</summary>

```
DATASET_REGISTRY["imagenet100"]["hf_id"]           = "clane9/imagenet-100"  # SUPERSEDED, unused
IMAGENET100_SYNSETS  = <CMC/Tian-et-al. list, now IMAGENET100_SYNSETS_CMC_ALTERNATIVE in source>
```
</details>

---

## Agent changes (Part U)

- **`vit_lora_cifar100_full5step_n5.py`** (the canonical source, restored + extended — not a
  separate script, per the "one source of truth" requirement):
  - Part A: `SEED` default restored to `42` (external `REPLICATION_SEED` override preserved).
  - Part B: `RUN_NAME_BASE` restored to a neutral, parameterized form
    (`DATASET_NAME`/`EXPERIMENT_LABEL`/`SEED` composed, no hardcoded "cifar100" or "seed123"
    literal).
  - Part C: verified (no change needed) `USE_RANKEXT_RANK_SCHEDULE_WIDE=False`,
    `RANKEXT_RANK_SCHEDULE=[20,40,60,80]`.
  - New `DATASET_REGISTRY` dict + `IMAGENET100_SYNSETS` list (with provenance/caveat comments) +
    a consolidated experiment-invariant assertion block (dataset name, class/step counts, backbone,
    RankExt schedule, stale-naming guards for the imagenet100 path).
  - `load_dataset("cifar100")` → dispatches on `DATASET_REGISTRY["loader"]` (`"hf_hub"` for
    cifar100, unchanged; `"local_imagefolder"` for imagenet100, reading the shared
    `IMAGENET_ROOT`); loader-time class-set assertion added for the imagenet100 path.
  - `dataset["test"]` in `make_eval_dataset()` → `dataset[DATASET_REGISTRY["final_eval_split"]]`.
  - **UPDATE (Section 0, same day):** `IMAGENET100_SYNSETS` **replaced** with the verified
    PODNet/DER/DyTox lineage list (was the CMC list, now preserved inactive as
    `IMAGENET100_SYNSETS_CMC_ALTERNATIVE`); added `IMAGENET100_CLASS_ORDER` (seed-42 permutation)
    and `IMAGENET100_WNID_TO_LABEL`; loader switched from the Hugging Face Hub mirror to a local
    `imagefolder` load against `IMAGENET_ROOT` (env-var, default
    `/nfsd/lttm4/datasets/ImageNet-1k_torch`), filtered to the 100 selected WNIDs via `data_files`
    globs, with an explicit label-remap step (`_remap_imagenet100_label`) so the loader's own
    ClassLabel indices never leak into the classifier unremapped.
  - **Not changed:** any SimpleAvg/RankExt/KD/FactorOrth/calibration/metric/warmup/anchor/protection
    code, `MODEL_CHECKPOINT`, `NUM_CLASSES`/`NUM_STEPS`/`CLASSES_PER_STEP`, the transform pipeline,
    `build_classwise_train_val_splits()`, Chapter 3 methodology, Chapter 5/6 prose.
- **`scripts/verify_imagenet100_local.py`** (new) — standalone, read-only local-dataset
  verification utility (Section 0.5); sandbox-tested, not yet run against the real cluster path.
- **`thesis_agent/reports/imagenet100_generalization_preparation.md`** — this report (Section 0
  added same day as the original write-up, after the benchmark-definition deep-audit).
- **`thesis_agent/CONTINUE_LATER.md`** — appended a new status section (planned, not launched;
  pointer to this report) and updated the closing "NEXT TASK" line to list this alongside the
  pre-existing, still-outstanding documentation-refinement queue, without reordering or
  deprioritizing either.
- **Deliberately NOT touched:** `data/experiments.jsonl`, `data/claims.jsonl`, `data/decisions.jsonl`
  (no experiment has actually run, so there is no result/claim/decision-with-an-outcome to record
  yet — adding a placeholder row would violate `build_knowledge_base.py`'s own stated policy of not
  adding entries without a citable outcome); `chapters/chapter_3_methodology.md`; any file under
  `thesis_agent/chapters/` describing Chapter 5/6 content; any file under
  `thesis_writing/thesis_latex/chapters/`.
- `validate_knowledge_base.py` was **not** re-run for this task, since no `data/*.jsonl` file was
  regenerated (nothing changed in `build_knowledge_base.py`).
