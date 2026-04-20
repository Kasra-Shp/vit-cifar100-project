#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.environ.setdefault("HF_HOME", "/nfsd/lttm4/tesisti/shahrampour/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/nfsd/lttm4/tesisti/shahrampour/hf_datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/nfsd/lttm4/tesisti/shahrampour/hf_transformers")

for k in ["HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE"]:
    os.makedirs(os.environ[k], exist_ok=True)

print("HF_HOME:", os.environ["HF_HOME"])
print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
print("TRANSFORMERS_CACHE:", os.environ["TRANSFORMERS_CACHE"])


# ## 1) Imports

# In[ ]:


import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import copy
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file as safe_load_file
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[ ]:


RUN_NAME = "cifar100_5step_orth_lambda_sweep_quick_n6"

RESULTS_DIR = os.path.join("results", RUN_NAME)
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ## 2) Load CIFAR-100 (fine labels)

# In[ ]:


from datasets import Image

dataset = load_dataset("cifar100")
dataset = dataset.rename_column("fine_label", "label")

dataset = dataset.cast_column("img", Image())

class_names = dataset["train"].features["label"].names
num_classes = len(class_names)

print("num_classes:", num_classes)
print("example keys:", dataset["train"][0].keys())
print("first 10 classes:", class_names[:10])


# In[ ]:


def make_custom_eval_dataset(class_ids, remap_labels=True):
    test_ds = filter_dataset_by_classes(dataset["test"], class_ids)

    if remap_labels:
        label2new = {orig: i for i, orig in enumerate(sorted(class_ids))}
        new2orig = {v: k for k, v in label2new.items()}

        def remap(ex):
            ex["label"] = label2new[int(ex["label"])]
            return ex

        test_ds = test_ds.map(remap)
    else:
        label2new = None
        new2orig = None

    test_ds.set_transform(preprocess_val)
    return test_ds, label2new, new2orig


# ## 3) Define incremental class splits (2/5/10 steps)

# In[ ]:


num_steps = 5  

assert num_classes % num_steps == 0
classes_per_step = num_classes // num_steps

class_splits = [
    list(range(s * classes_per_step, (s + 1) * classes_per_step))
    for s in range(num_steps)
]

print("classes per step:", classes_per_step)
print("split sizes:", [len(x) for x in class_splits])
print("step0 sample:", class_splits[0][:10], "...", class_splits[0][-3:])


# In[ ]:


first_step_classes = class_splits[0]
later_step_classes = []
for s in range(1, num_steps):
    later_step_classes.extend(class_splits[s])
all_classes = list(range(num_classes))

print("first step classes:", len(first_step_classes))
print("later step classes:", len(later_step_classes))
print("all classes:", len(all_classes))


# ## 4) Model + preprocessing

# In[ ]:


# Requested model
model_checkpoint = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

from torchvision import transforms
from PIL import Image
import numpy as np
import torch

size = image_processor.size
if isinstance(size, dict):
    H = size.get("height", 224)
    W = size.get("width", 224)
else:
    H = W = int(size)

train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomCrop((H, W), padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

val_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

def to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, dict):
        if "array" in x:
            x = x["array"]
        elif "bytes" in x:
            import io
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")

    if isinstance(x, list):
        x = np.array(x, dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = x.astype(np.uint8)
        arr = np.squeeze(arr)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if not (arr.ndim == 3 and arr.shape[-1] in (1, 3)):
            raise TypeError(f"Unexpected image array shape after squeeze: {arr.shape}")

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return Image.fromarray(arr).convert("RGB")

    return x

def preprocess_train(ex):
    ex["pixel_values"] = [train_transform(to_pil(img)) for img in ex["img"]]
    ex["labels"] = ex["label"]
    return ex

def preprocess_val(ex):
    ex["pixel_values"] = [val_transform(to_pil(img)) for img in ex["img"]]
    ex["labels"] = ex["label"]
    return ex

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([int(e["labels"]) for e in examples], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean().item()}


# ## 5) Build per-step datasets (step / cumulative / full)

# In[ ]:


def classes_for_step(step_idx: int) -> list[int]:
    return class_splits[step_idx]

def classes_for_cumulative(step_idx: int) -> list[int]:
    out = []
    for s in range(step_idx + 1):
        out.extend(class_splits[s])
    return out

def filter_by_classes(ds, class_ids):
    class_set = set(class_ids)
    ds = ds.with_format(None)
    return ds.filter(lambda x: int(x["label"]) in class_set)

def make_step_datasets(step_idx: int, split_type: str = "new_only", remap_labels: bool = False):
    """
    split_type:
      - 'new_only'   : only classes of this step
      - 'cumulative' : union of classes up to this step
      - 'full'       : all classes (100)
    """
    if split_type == "full":
        class_ids = list(range(num_classes))
    elif split_type == "cumulative":
        class_ids = classes_for_cumulative(step_idx)
    elif split_type == "new_only":
        class_ids = classes_for_step(step_idx)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    train_ds = filter_by_classes(dataset["train"], class_ids)
    test_ds  = filter_by_classes(dataset["test"], class_ids)

    if remap_labels:
        label2new = {orig: i for i, orig in enumerate(sorted(class_ids))}
        new2orig = {v: k for k, v in label2new.items()}

        def remap(ex):
            ex["label"] = label2new[int(ex["label"])]
            return ex

        train_ds = train_ds.map(remap)
        test_ds  = test_ds.map(remap)
    else:
        label2new = {c: c for c in class_ids}
        new2orig = {c: c for c in class_ids}

    train_ds = train_ds.with_transform(preprocess_train)
    test_ds = test_ds.with_transform(preprocess_val)

    print(f"[Dataset] Step {step_idx+1} | split_type={split_type}")
    print(f"[Dataset] Classes: {class_ids[:5]} ... {class_ids[-5:]}")
    print(f"[Dataset] Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    return train_ds, test_ds, label2new, new2orig, class_ids

eval_all = dataset["test"].with_transform(preprocess_val)

print("eval_all size:", len(eval_all))


# ## 6) Training recipes (reasonable settings)

# In[ ]:


set_seed(42)

SCRATCH_EPOCHS = 8
LORA_EPOCHS    = 1
FT_EPOCHS      = 1
JOINT_EPOCHS   = 1

BATCH_SCRATCH = 8
ACCUM_SCRATCH = 2

BATCH_LORA    = 16
ACCUM_LORA    = 1

BATCH_FT      = 8
ACCUM_FT      = 2

LR_SCRATCH = 5e-5
LR_LORA    = 1.5e-5
LR_FT      = 3e-5
LR_JOINT   = 5e-5

WEIGHT_DECAY = 0.05
WEIGHT_DECAY_LORA = 0.05
WARMUP_RATIO = 0.10
SCHED = "cosine"

USE_FP16 = torch.cuda.is_available()

results = []


# In[ ]:


def lambda_tag(x):
    s = f"{x:.0e}" if x < 1e-2 and x > 0 else str(x)
    return s.replace("-", "m").replace(".", "p")

run_config = {
    "run_name": RUN_NAME,
    "model_checkpoint": model_checkpoint,
    "init_mode": "scratch",
    "num_classes": num_classes,
    "num_steps": num_steps,
    "classes_per_step": classes_per_step,
    "scratch_epochs": SCRATCH_EPOCHS,
    "lora_epochs": LORA_EPOCHS,
    "ft_epochs": FT_EPOCHS,
    "joint_epochs": JOINT_EPOCHS,
    "lr_scratch": LR_SCRATCH,
    "lr_lora": LR_LORA,
    "lr_ft": LR_FT,
    "lr_joint": LR_JOINT,

    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],

    "variants": ["orth"],
    "orth_lambda_sweep": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
}

with open(os.path.join(METRICS_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

LORA_R = run_config["lora_r"]
LORA_ALPHA = run_config["lora_alpha"]
LORA_DROPOUT = run_config["lora_dropout"]
TARGET_MODULES = run_config["target_modules"]

VARIANTS = run_config["variants"]
ORTH_LAMBDA_SWEEP = run_config["orth_lambda_sweep"]


# In[ ]:


OUTPUTS_ROOT = os.path.join("outputs", RUN_NAME)
os.makedirs(OUTPUTS_ROOT, exist_ok=True)

STEP1_OUT = os.path.join(OUTPUTS_ROOT, "step1_scratch")
STEP1_FINAL_OUT = os.path.join(OUTPUTS_ROOT, "step1_scratch_best")

LORA_BANK_DIR = os.path.join(OUTPUTS_ROOT, "adapter_bank")
JOINT_OUT = os.path.join(OUTPUTS_ROOT, "joint")

os.makedirs(LORA_BANK_DIR, exist_ok=True)
os.makedirs(JOINT_OUT, exist_ok=True)

# compatibility placeholders for old cells
LAMBDA_ORTH = 0.0
merged_final_evals = {}


# ## 7) Step 1: train full ViT from scratch on step 0 classes

# In[ ]:


import os, shutil, json

# --- clean old step1 outputs so stale checkpoints cannot survive ---
if os.path.exists(STEP1_OUT):
    shutil.rmtree(STEP1_OUT)
if os.path.exists(STEP1_FINAL_OUT):
    shutil.rmtree(STEP1_FINAL_OUT)

os.makedirs(STEP1_OUT, exist_ok=True)
os.makedirs(STEP1_FINAL_OUT, exist_ok=True)

step1_idx = 0
train_step1, test_step1, label2new_1, new2orig_1, class_ids_1 = make_step_datasets(
    step1_idx, split_type="new_only", remap_labels=False
)

print("Step1 original classes:", class_ids_1[:5], "...", class_ids_1[-5:])
print(
    "Step1 label range:",
    min(int(train_step1[i]["label"]) for i in range(min(200, len(train_step1)))),
    max(int(train_step1[i]["label"]) for i in range(min(200, len(train_step1)))),
)

config_step1 = AutoConfig.from_pretrained(
    model_checkpoint,
    num_labels=num_classes,
    id2label={i: str(i) for i in range(num_classes)},
    label2id={str(i): i for i in range(num_classes)},
)

model_step1 = AutoModelForImageClassification.from_config(config_step1)

print("Before training - Step1 model num_labels:", model_step1.config.num_labels)
print("Before training - Step1 classifier out_features:", model_step1.classifier.out_features)
assert model_step1.config.num_labels == num_classes
assert model_step1.classifier.out_features == num_classes

args_step1 = TrainingArguments(
    output_dir=STEP1_OUT,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=SCRATCH_EPOCHS,
    learning_rate=LR_SCRATCH,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=SCHED,
    per_device_train_batch_size=BATCH_SCRATCH,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=ACCUM_SCRATCH,
    fp16=USE_FP16,
    dataloader_num_workers=4,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    disable_tqdm=True,
    max_grad_norm=1.0,
)

trainer_step1 = Trainer(
    model=model_step1,
    args=args_step1,
    train_dataset=train_step1,
    eval_dataset=test_step1,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

from transformers.utils.notebook import NotebookProgressCallback
trainer_step1.remove_callback(NotebookProgressCallback)

trainer_step1.train()

eval_step1 = trainer_step1.evaluate()
print({"eval_step1": eval_step1})

print("About to save B0 final model to:", STEP1_FINAL_OUT)
print("Trainer model type:", type(trainer_step1.model))
print("Trainer model class:", trainer_step1.model.__class__.__name__)

trainer_step1.model.save_pretrained(STEP1_FINAL_OUT, safe_serialization=False)
image_processor.save_pretrained(STEP1_FINAL_OUT)

print("Saved STEP1_FINAL_OUT to:", STEP1_FINAL_OUT)
print("STEP1_FINAL_OUT exists:", os.path.exists(STEP1_FINAL_OUT))
print("Files in STEP1_FINAL_OUT:", os.listdir(STEP1_FINAL_OUT) if os.path.exists(STEP1_FINAL_OUT) else "MISSING")

cfg_path = os.path.join(STEP1_FINAL_OUT, "config.json")
print("config exists:", os.path.exists(cfg_path))

if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    print("model_type:", cfg.get("model_type"))
    print("architectures:", cfg.get("architectures"))
    print("num_labels:", cfg.get("num_labels"))

reloaded_step1 = AutoModelForImageClassification.from_pretrained(STEP1_FINAL_OUT)
print("Reloaded STEP1 num_labels:", reloaded_step1.config.num_labels)
print("Reloaded STEP1 classifier out_features:", reloaded_step1.classifier.out_features)

assert reloaded_step1.config.num_labels == num_classes
assert reloaded_step1.classifier.out_features == num_classes


# In[ ]:


step1_model_check = AutoModelForImageClassification.from_pretrained(STEP1_FINAL_OUT)

print("num_labels:", step1_model_check.config.num_labels)
print("classifier out_features:", step1_model_check.classifier.out_features)
assert step1_model_check.config.num_labels == num_classes
assert step1_model_check.classifier.out_features == num_classes


# In[ ]:


print("Init mode:", run_config["init_mode"])
assert run_config["init_mode"] == "scratch"


# In[ ]:


step1_log_df = pd.DataFrame(trainer_step1.state.log_history)
step1_log_df.to_csv(os.path.join(TABLES_DIR, "step1_log_history.csv"), index=False)

step1_metrics = {
    "experiment": "step1_scratch",
    "eval_loss": float(eval_step1.get("eval_loss", np.nan)),
    "eval_accuracy": float(eval_step1.get("eval_accuracy", np.nan)),
}

with open(os.path.join(METRICS_DIR, "step1_metrics.json"), "w") as f:
    json.dump(step1_metrics, f, indent=2)

results.append({
    "experiment": "step1_scratch",
    "method": "step1_scratch",
    "step": 1,
    "eval_type": "current_step",
    "eval_accuracy": float(eval_step1.get("eval_accuracy", np.nan)),
    "eval_loss": float(eval_step1.get("eval_loss", np.nan)),
})

step1_log_df.tail()


# In[ ]:


if "loss" in step1_log_df.columns:
    train_loss_df = step1_log_df[step1_log_df["loss"].notna()]
    if not train_loss_df.empty:
        plt.figure(figsize=(8,5))
        plt.plot(train_loss_df["step"], train_loss_df["loss"])
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Step 1 Train Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "step1_train_loss.png"), dpi=200)
        plt.show()

if "eval_accuracy" in step1_log_df.columns:
    eval_df = step1_log_df[step1_log_df["eval_accuracy"].notna()]
    if not eval_df.empty:
        plt.figure(figsize=(8,5))
        plt.plot(eval_df["epoch"], eval_df["eval_accuracy"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Eval Accuracy")
        plt.title("Step 1 Eval Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "step1_eval_accuracy.png"), dpi=200)
        plt.show()


# In[ ]:


def get_module_by_name(root_module, dotted_name):
    cur = root_module
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def get_classifier_module(model):
    candidate_paths = [
        "classifier",
        "base_model.classifier",
        "base_model.model.classifier",
        "model.classifier",
    ]
    for path in candidate_paths:
        mod = get_module_by_name(model, path)
        if mod is not None and hasattr(mod, "weight"):
            return mod, path
    raise ValueError("Could not find classifier module in model.")



def install_classifier_row_mask(model, allowed_rows):
    clf = model.classifier

    device = clf.weight.device

    weight_mask = torch.zeros_like(clf.weight, device=device)
    bias_mask = torch.zeros_like(clf.bias, device=device)

    weight_mask[allowed_rows, :] = 1.0
    bias_mask[allowed_rows] = 1.0

    weight_handle = clf.weight.register_hook(
        lambda grad, mask=weight_mask: grad * mask.to(grad.device)
    )

    bias_handle = clf.bias.register_hook(
        lambda grad, mask=bias_mask: grad * mask.to(grad.device)
    )

    return {
        "weight_handle": weight_handle,
        "bias_handle": bias_handle,
    }
   

def remove_classifier_row_mask(mask_state):
    if mask_state is None:
        return
    if mask_state.get("weight_handle", None) is not None:
        mask_state["weight_handle"].remove()
    if mask_state.get("bias_handle", None) is not None:
        mask_state["bias_handle"].remove()


def print_trainable_summary(model):
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(x[1] for x in trainable)
    print(f"[Trainable summary] tensors: {len(trainable)} | params: {total_trainable:,}")
    print("[Trainable summary] classifier trainable tensors:")
    for n, sz in trainable:
        if "classifier" in n:
            print("   ", n, sz)


# ## 8) Step 2: LoRA only (freeze backbone) on top of Step 1 model

# In[ ]:


first_step_classes = classes_for_step(0)

def make_eval_dataset(class_ids, name=None):
    test_ds = filter_by_classes(dataset["test"], class_ids)
    test_ds = test_ds.with_transform(preprocess_val)
    if name is not None:
        print(f"[Eval set] {name}: num_classes={len(class_ids)}, size={len(test_ds)}")
    return test_ds


# In[ ]:


def inspect_dataset_labels(ds, name, n=300):
    vals = [int(ds[i]["label"]) for i in range(min(n, len(ds)))]
    print(f"{name}:")
    print("  size =", len(ds))
    print("  min/max =", min(vals), max(vals))
    print("  unique(sample) =", sorted(set(vals))[:50])

for step_idx in range(num_steps):
    tr, te, _, _, class_ids = make_step_datasets(step_idx, split_type="new_only", remap_labels=False)
    inspect_dataset_labels(tr, f"train step {step_idx+1}")
    inspect_dataset_labels(te, f"test  step {step_idx+1}")
    print("  expected classes:", class_ids)
    assert set(sorted(set(int(tr[i]['label']) for i in range(min(len(tr), 500))))) <= set(class_ids)
    print("-" * 80)


# In[ ]:


def normalize_module_name(module_name):
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.",
    ]
    for p in prefixes:
        if module_name.startswith(p):
            module_name = module_name[len(p):]
    module_name = module_name.replace("vit.model.", "vit.")
    return module_name


def load_adapter_state(adapter_dir):
    safe_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        return load_file(safe_path)
    elif os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def get_lora_scaling_factor(adapter_dir):
    from peft import PeftConfig
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    return peft_cfg.lora_alpha / peft_cfg.r


def extract_lora_factor_matrices(adapter_state):
    lora_factor_matrices = {}

    for param_name, param_tensor in adapter_state.items():
        if ".lora_A." in param_name:
            base_module_name = normalize_module_name(param_name.split(".lora_A.")[0])
            lora_factor_matrices.setdefault(base_module_name, {})["A_matrix"] = param_tensor.detach().cpu().float()

        elif ".lora_B." in param_name:
            base_module_name = normalize_module_name(param_name.split(".lora_B.")[0])
            lora_factor_matrices.setdefault(base_module_name, {})["B_matrix"] = param_tensor.detach().cpu().float()

    lora_factor_matrices = {
        module_name: factors
        for module_name, factors in lora_factor_matrices.items()
        if "A_matrix" in factors and "B_matrix" in factors
    }

    return lora_factor_matrices


def sample_replay_from_previous_steps(step_idx, per_old_step=250, seed=42):
    replay_parts = []

    for prev_step_idx in range(step_idx):
        prev_train, _, _, _, _ = make_step_datasets(
            prev_step_idx,
            split_type="new_only",
            remap_labels=False,
        )
        n_take = min(per_old_step, len(prev_train))
        replay_part = prev_train.shuffle(seed=seed + prev_step_idx).select(range(n_take))
        replay_parts.append(replay_part)

    if len(replay_parts) == 0:
        return None

    return concatenate_datasets(replay_parts)


def build_variant_train_dataset(step_idx, variant_name):
    train_step, test_step, label2new, new2orig, class_ids = make_step_datasets(
        step_idx,
        split_type="new_only",
        remap_labels=False,
    )

    if variant_name == "replay":
        raise ValueError("Replay variant is disabled in this notebook.")

    return train_step, test_step, label2new, new2orig, class_ids


def get_submodule_by_name(model, module_name):
    module_name = normalize_module_name(module_name)
    submodule = model
    for part in module_name.split("."):
        submodule = getattr(submodule, part)
    return submodule


def build_prev_weight_bank_from_model(model):
    weight_bank = {}

    for module_name, module in model.named_modules():
        norm_name = normalize_module_name(module_name)
        if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
            if any(t in norm_name for t in TARGET_MODULES):
                weight_bank[norm_name] = module.weight.detach().cpu().float().clone()

    return weight_bank


def compute_exact_orth_penalty_from_model(model, prev_weight_bank):
    penalties = []

    for module_name, module in model.named_modules():
        norm_name = normalize_module_name(module_name)

        if norm_name not in prev_weight_bank:
            continue

        has_lora = (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        )
        if not has_lora:
            continue

        A = module.lora_A["default"].weight
        B = module.lora_B["default"].weight

        if hasattr(module, "scaling"):
            scaling = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
        else:
            scaling = 1.0

        curr_delta = (B @ A) * float(scaling)   # L_t = ΔW = AB
        prev_W = prev_weight_bank[norm_name].to(curr_delta.device, dtype=curr_delta.dtype)  # M_{t-1}

        # tr(M_(t-1) L_t^T)
        trace_term = torch.trace(prev_W @ curr_delta.T)
        penalties.append(trace_term)

    if len(penalties) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    return torch.stack(penalties).mean()




class ExactOrthogonalLoRATrainer(Trainer):
    def __init__(self, *args, lambda_orth=0.0, ortho_debug_every=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_orth = lambda_orth
        self.ortho_debug_every = ortho_debug_every

    def _compute_orth_penalty(self, model):
        penalties = []
        adapter_name = "default"

        for name, module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue
            if not hasattr(module, "base_layer") or not hasattr(module.base_layer, "weight"):
                continue

            A = module.lora_A[adapter_name].weight
            B = module.lora_B[adapter_name].weight

            if isinstance(module.scaling, dict):
                scaling = module.scaling[adapter_name]
            else:
                scaling = module.scaling

            delta_w = (B @ A) * scaling

            if getattr(module, "fan_in_fan_out", False):
                delta_w = delta_w.T

            W_prev = module.base_layer.weight.detach()

            inner = torch.sum(W_prev * delta_w)
            denom = (W_prev.norm(p="fro") * delta_w.norm(p="fro")).clamp_min(1e-12)
            penalty = (inner / denom).pow(2)

            penalties.append(penalty)

        if len(penalties) == 0:
            return torch.zeros((), device=next(model.parameters()).device)

        return torch.stack(penalties).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        ortho_loss = self._compute_orth_penalty(model)
        weighted_ortho = self.lambda_orth * ortho_loss
        loss = ce_loss + weighted_ortho

        if self.state.global_step % self.ortho_debug_every == 0:
            ratio = (weighted_ortho / ce_loss.clamp_min(1e-12)).item()
            print(
                f"[ORTH DEBUG] step={self.state.global_step} "
                f"ce={ce_loss.item():.6f} "
                f"ortho={ortho_loss.item():.6f} "
                f"weighted_ortho={weighted_ortho.item():.6f} "
                f"lambda={self.lambda_orth:.6f} "
                f"ratio={ratio:.8f} "
                f"total={loss.item():.6f}"
            )

        if return_outputs:
            return loss, outputs
        return loss

def extract_classifier_tensors(adapter_state):
    classifier_weight_key = None
    classifier_bias_key = None

    for param_name in adapter_state.keys():
        if "classifier" in param_name and param_name.endswith("weight"):
            classifier_weight_key = param_name
        if "classifier" in param_name and param_name.endswith("bias"):
            classifier_bias_key = param_name

    if classifier_weight_key is None:
        return None, None

    classifier_weight_tensor = adapter_state[classifier_weight_key].detach().cpu().float()
    classifier_bias_tensor = (
        adapter_state[classifier_bias_key].detach().cpu().float()
        if classifier_bias_key is not None else None
    )

    return classifier_weight_tensor, classifier_bias_tensor


def build_fixed_avg_merged_model(base_model_dir, adapter_dirs, adapter_class_groups):
    base_model_W0 = AutoModelForImageClassification.from_pretrained(base_model_dir)
    base_model_W0.eval()

    module_to_delta_list = {}
    classifier_row_sources = []

    for adapter_dir, class_ids_for_this_adapter in zip(adapter_dirs, adapter_class_groups):
        adapter_state = load_adapter_state(adapter_dir)
        lora_scaling_factor_si = get_lora_scaling_factor(adapter_dir)
        module_to_factor_matrices = extract_lora_factor_matrices(adapter_state)

        for module_name, factor_dict in module_to_factor_matrices.items():
            A_i = factor_dict["A_matrix"]
            B_i = factor_dict["B_matrix"]
            effective_lora_delta = (B_i @ A_i) * lora_scaling_factor_si
            module_to_delta_list.setdefault(module_name, []).append(effective_lora_delta)

        classifier_weight_i, classifier_bias_i = extract_classifier_tensors(adapter_state)
        classifier_row_sources.append(
            (class_ids_for_this_adapter, classifier_weight_i, classifier_bias_i)
        )

    with torch.no_grad():
        for module_name, delta_list in module_to_delta_list.items():
            avg_delta = torch.stack(delta_list, dim=0).mean(dim=0)
            target_module = get_submodule_by_name(base_model_W0, module_name)

            if target_module.weight.shape != avg_delta.shape:
                raise ValueError(
                    f"Shape mismatch for {module_name}: "
                    f"weight={target_module.weight.shape}, delta={avg_delta.shape}"
                )

            target_module.weight.data += avg_delta.to(
                target_module.weight.device,
                dtype=target_module.weight.dtype
            )

    with torch.no_grad():
        for class_ids_for_this_adapter, classifier_weight_i, classifier_bias_i in classifier_row_sources:
            if classifier_weight_i is None:
                continue

            base_model_W0.classifier.weight.data[class_ids_for_this_adapter] = classifier_weight_i[class_ids_for_this_adapter].to(
                base_model_W0.classifier.weight.device,
                dtype=base_model_W0.classifier.weight.dtype
            )

            if classifier_bias_i is not None and base_model_W0.classifier.bias is not None:
                base_model_W0.classifier.bias.data[class_ids_for_this_adapter] = classifier_bias_i[class_ids_for_this_adapter].to(
                    base_model_W0.classifier.bias.device,
                    dtype=base_model_W0.classifier.bias.dtype
                )

    base_model_W0.eval()
    return base_model_W0


# In[ ]:


def get_submodule_by_name(model, module_name):
    module_name = normalize_module_name(module_name)
    submodule = model
    for part in module_name.split("."):
        submodule = getattr(submodule, part)
    return submodule


def extract_classifier_tensors(adapter_state):
    classifier_weight_key = None
    classifier_bias_key = None

    for param_name in adapter_state.keys():
        if "classifier" in param_name and param_name.endswith("weight"):
            classifier_weight_key = param_name
        if "classifier" in param_name and param_name.endswith("bias"):
            classifier_bias_key = param_name

    if classifier_weight_key is None:
        return None, None

    classifier_weight_tensor = adapter_state[classifier_weight_key].detach().cpu().float()
    classifier_bias_tensor = (
        adapter_state[classifier_bias_key].detach().cpu().float()
        if classifier_bias_key is not None else None
    )

    return classifier_weight_tensor, classifier_bias_tensor


def build_fixed_avg_merged_model(base_model_dir, adapter_dirs, adapter_class_groups):
    base_model_W0 = AutoModelForImageClassification.from_pretrained(base_model_dir)
    base_model_W0.eval()

    module_to_delta_list = {}
    classifier_row_sources = []

    for adapter_dir, class_ids_for_this_adapter in zip(adapter_dirs, adapter_class_groups):
        adapter_state = load_adapter_state(adapter_dir)
        lora_scaling_factor_si = get_lora_scaling_factor(adapter_dir)
        module_to_factor_matrices = extract_lora_factor_matrices(adapter_state)

        for module_name, factor_dict in module_to_factor_matrices.items():
            A_i = factor_dict["A_matrix"]
            B_i = factor_dict["B_matrix"]
            effective_lora_delta = (B_i @ A_i) * lora_scaling_factor_si
            module_to_delta_list.setdefault(module_name, []).append(effective_lora_delta)

        classifier_weight_i, classifier_bias_i = extract_classifier_tensors(adapter_state)
        classifier_row_sources.append(
            (class_ids_for_this_adapter, classifier_weight_i, classifier_bias_i)
        )

    with torch.no_grad():
        for module_name, delta_list in module_to_delta_list.items():
            avg_delta = torch.stack(delta_list, dim=0).mean(dim=0)
            target_module = get_submodule_by_name(base_model_W0, module_name)

            if target_module.weight.shape != avg_delta.shape:
                raise ValueError(
                    f"Shape mismatch for {module_name}: "
                    f"weight={target_module.weight.shape}, delta={avg_delta.shape}"
                )

            target_module.weight.data += avg_delta.to(
                target_module.weight.device,
                dtype=target_module.weight.dtype
            )

    with torch.no_grad():
        for class_ids_for_this_adapter, classifier_weight_i, classifier_bias_i in classifier_row_sources:
            if classifier_weight_i is None:
                continue

            base_model_W0.classifier.weight.data[class_ids_for_this_adapter] = classifier_weight_i[class_ids_for_this_adapter].to(
                base_model_W0.classifier.weight.device,
                dtype=base_model_W0.classifier.weight.dtype
            )

            if classifier_bias_i is not None and base_model_W0.classifier.bias is not None:
                base_model_W0.classifier.bias.data[class_ids_for_this_adapter] = classifier_bias_i[class_ids_for_this_adapter].to(
                    base_model_W0.classifier.bias.device,
                    dtype=base_model_W0.classifier.bias.dtype
                )

    base_model_W0.eval()
    return base_model_W0


# In[ ]:


lora_results = []

first_step_classes = classes_for_step(0)

def _label_range(ds, n=200):
    vals = [int(ds[i]["label"]) for i in range(min(n, len(ds)))]
    return min(vals), max(vals)

ORTH_SWEEP_ROOT = os.path.join("outputs", RUN_NAME, "orth_lambda_sweep")
os.makedirs(ORTH_SWEEP_ROOT, exist_ok=True)

for lambda_orth in ORTH_LAMBDA_SWEEP:
    lam_tag = lambda_tag(lambda_orth)

    print("\n" + "=" * 100)
    print(f"ORTH SWEEP | lambda_orth = {lambda_orth}")
    print("=" * 100)

    variant_bank_dir = os.path.join(ORTH_SWEEP_ROOT, f"lambda_{lam_tag}", "adapter_bank")
    orth_chain_dir   = os.path.join(ORTH_SWEEP_ROOT, f"lambda_{lam_tag}", "orth_chain")

    if os.path.exists(variant_bank_dir):
        shutil.rmtree(variant_bank_dir)
    if os.path.exists(orth_chain_dir):
        shutil.rmtree(orth_chain_dir)

    os.makedirs(variant_bank_dir, exist_ok=True)
    os.makedirs(orth_chain_dir, exist_ok=True)

    orth_current_base_dir = STEP1_FINAL_OUT

    for step_idx in range(1, num_steps):
        train_step, test_step, label2new, new2orig, class_ids = build_variant_train_dataset(
            step_idx,
            "orth",
        )

        print(f"\n[orth | lambda={lambda_orth}] Step {step_idx+1}")
        print("Current step classes:", class_ids[:5], "...", class_ids[-5:])
        print("Train size:", len(train_step), "| Test size:", len(test_step))

        tr_min, tr_max = _label_range(train_step)
        te_min, te_max = _label_range(test_step)

        print("Train label range:", tr_min, tr_max)
        print("Test label range :", te_min, te_max)

        base_model_dir = orth_current_base_dir
        base_model = AutoModelForImageClassification.from_pretrained(base_model_dir)

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=TARGET_MODULES,
            modules_to_save=["classifier"],
        )

        model_lora = get_peft_model(base_model, lora_config)

        classifier_mask_state = install_classifier_row_mask(
            model_lora,
            allowed_rows=class_ids
        )

        print_trainable_summary(model_lora)

        step_train_dir = os.path.join(
            ORTH_SWEEP_ROOT,
            f"lambda_{lam_tag}",
            f"step_{step_idx+1}_lora_train"
        )

        args_lora = TrainingArguments(
            output_dir=step_train_dir,
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            num_train_epochs=LORA_EPOCHS,
            learning_rate=LR_LORA,
            weight_decay=WEIGHT_DECAY_LORA,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=SCHED,
            per_device_train_batch_size=BATCH_LORA,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=ACCUM_LORA,
            fp16=USE_FP16,
            dataloader_num_workers=4,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            max_grad_norm=1.0,
        )

        trainer_lora = ExactOrthogonalLoRATrainer(
            model=model_lora,
            args=args_lora,
            train_dataset=train_step,
            eval_dataset=test_step,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            lambda_orth=lambda_orth,
            ortho_debug_every=50,
        )

        trainer_lora.train()

        eval_current = trainer_lora.evaluate(eval_dataset=test_step)

        eval_first = make_eval_dataset(classes_for_step(0))
        seen_now = classes_for_cumulative(step_idx)
        later_now = [c for c in seen_now if c not in classes_for_step(0)]

        eval_later = make_eval_dataset(later_now) if len(later_now) > 0 else None
        eval_seen  = make_eval_dataset(seen_now)

        eval_first_res = trainer_lora.evaluate(eval_dataset=eval_first)
        eval_later_res = trainer_lora.evaluate(eval_dataset=eval_later) if eval_later is not None else None
        eval_seen_res  = trainer_lora.evaluate(eval_dataset=eval_seen)

        lora_results.append({
            "method": "LoRA_orth_rowmask",
            "lambda_orth": lambda_orth,
            "step": step_idx + 1,
            "eval_set": "current_step",
            **eval_current,
        })
        lora_results.append({
            "method": "LoRA_orth_rowmask",
            "lambda_orth": lambda_orth,
            "step": step_idx + 1,
            "eval_set": "first_step",
            **eval_first_res,
        })
        if eval_later_res is not None:
            lora_results.append({
                "method": "LoRA_orth_rowmask",
                "lambda_orth": lambda_orth,
                "step": step_idx + 1,
                "eval_set": "later_steps",
                **eval_later_res,
            })
        lora_results.append({
            "method": "LoRA_orth_rowmask",
            "lambda_orth": lambda_orth,
            "step": step_idx + 1,
            "eval_set": "all_seen",
            **eval_seen_res,
        })

        adapter_save_dir = os.path.join(variant_bank_dir, f"step_{step_idx+1}_adapter")
        trainer_lora.model.save_pretrained(adapter_save_dir)

        absorbed_model = trainer_lora.model.merge_and_unload()
        absorbed_dir = os.path.join(orth_chain_dir, f"step_{step_idx+1}_absorbed")
        absorbed_model.save_pretrained(absorbed_dir)
        orth_current_base_dir = absorbed_dir

        remove_classifier_row_mask(classifier_mask_state)

        print(f"[orth | lambda={lambda_orth}] Step {step_idx+1} | current_step:", eval_current)
        print(f"[orth | lambda={lambda_orth}] Step {step_idx+1} | first_step  :", eval_first_res)
        print(f"[orth | lambda={lambda_orth}] Step {step_idx+1} | later_steps :", eval_later_res)
        print(f"[orth | lambda={lambda_orth}] Step {step_idx+1} | all_seen    :", eval_seen_res)


# In[ ]:


def filter_dataset_by_classes(dataset, class_ids):
    class_ids = set(class_ids)
    return dataset.filter(lambda x: x["label"] in class_ids)


# ## merging method

# In[ ]:


print("\n===== FINAL EVALUATION: ORTH LAMBDA SWEEP =====")

orth_sweep_final_evals = {}
orth_sweep_rows = []

for lambda_orth in ORTH_LAMBDA_SWEEP:
    lam_tag = lambda_tag(lambda_orth)

    print("\n" + "-" * 100)
    print(f"FINAL ORTH | lambda_orth = {lambda_orth}")
    print("-" * 100)

    final_model_dir = os.path.join(
        ORTH_SWEEP_ROOT,
        f"lambda_{lam_tag}",
        "orth_chain",
        f"step_{num_steps}_absorbed"
    )

    final_model = AutoModelForImageClassification.from_pretrained(final_model_dir)

    first_step_classes = classes_for_step(0)
    later_step_classes = []
    for s in range(1, num_steps):
        later_step_classes.extend(classes_for_step(s))
    final_seen_classes = classes_for_cumulative(num_steps - 1)

    eval_first = make_eval_dataset(first_step_classes, name=f"orth_lambda_{lambda_orth}_first")
    eval_later = make_eval_dataset(later_step_classes, name=f"orth_lambda_{lambda_orth}_later")
    eval_seen  = make_eval_dataset(final_seen_classes, name=f"orth_lambda_{lambda_orth}_seen")

    trainer_eval = Trainer(
        model=final_model,
        args=TrainingArguments(
            output_dir=os.path.join(ORTH_SWEEP_ROOT, f"lambda_{lam_tag}", "eval_final"),
            remove_unused_columns=False,
            per_device_eval_batch_size=32,
            report_to="none",
        ),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    final_first = trainer_eval.evaluate(eval_dataset=eval_first)
    final_later = trainer_eval.evaluate(eval_dataset=eval_later)
    final_seen  = trainer_eval.evaluate(eval_dataset=eval_seen)

    orth_sweep_final_evals[lambda_orth] = {
        "first_step": final_first,
        "later_steps": final_later,
        "all_seen": final_seen,
    }

    orth_sweep_rows.extend([
        {
            "lambda_orth": lambda_orth,
            "eval_set": "first_step",
            "accuracy": float(final_first["eval_accuracy"]),
            "loss": float(final_first["eval_loss"]),
        },
        {
            "lambda_orth": lambda_orth,
            "eval_set": "later_steps",
            "accuracy": float(final_later["eval_accuracy"]),
            "loss": float(final_later["eval_loss"]),
        },
        {
            "lambda_orth": lambda_orth,
            "eval_set": "all_seen",
            "accuracy": float(final_seen["eval_accuracy"]),
            "loss": float(final_seen["eval_loss"]),
        },
    ])

    print(f"[orth | lambda={lambda_orth}] final - first_step :", final_first)
    print(f"[orth | lambda={lambda_orth}] final - later_steps:", final_later)
    print(f"[orth | lambda={lambda_orth}] final - all_seen   :", final_seen)

orth_sweep_df = pd.DataFrame(orth_sweep_rows)
orth_sweep_df.to_csv(os.path.join(TABLES_DIR, "orth_lambda_sweep_final.csv"), index=False)

print("\nORTH SWEEP FINAL TABLE")
print(orth_sweep_df.sort_values(["eval_set", "lambda_orth"]))


# In[ ]:


final_model_dir = os.path.join(
    ORTH_SWEEP_ROOT,
    f"lambda_{lambda_tag(0.0)}",
    "orth_chain",
    f"step_{num_steps}_absorbed"
)

dbg_model = AutoModelForImageClassification.from_pretrained(final_model_dir)
dbg_model = dbg_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
dbg_model.eval()

dbg_eval_first = make_eval_dataset(classes_for_step(0), name="dbg_first_step")
batch = collate_fn([dbg_eval_first[i] for i in range(64)])
batch = {k: v.to(next(dbg_model.parameters()).device) for k, v in batch.items()}

with torch.no_grad():
    logits = dbg_model(pixel_values=batch["pixel_values"]).logits
    preds = logits.argmax(dim=1).cpu()

print("True labels   :", batch["labels"][:30].cpu().tolist())
print("Pred labels   :", preds[:30].tolist())
print("Pred unique   :", sorted(set(preds.tolist()))[:50])
print("First-step range should be within:", classes_for_step(0))


# In[ ]:


pivot_df = orth_sweep_df.pivot(index="lambda_orth", columns="eval_set", values="accuracy")
pivot_df = pivot_df[["first_step", "later_steps", "all_seen"]]

print(pivot_df.sort_values("all_seen", ascending=False))

plt.figure(figsize=(10, 6))
for col in ["first_step", "later_steps", "all_seen"]:
    plt.plot(pivot_df.index, pivot_df[col], marker="o", label=col)

plt.xscale("log")
plt.xlabel("lambda_orth")
plt.ylabel("final accuracy")
plt.title("Orth lambda sweep (final accuracies)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "orth_lambda_sweep_final.png"))
plt.show()


# ## 9) Baseline: full fine-tune instead of LoRA (Step 2)

# In[ ]:


# ft_results = []

# for s in range(2, num_steps + 1):
#     stale_train_dir = f"outputs/{RUN_NAME}/step_{s}_ft"
#     stale_final_dir = f"outputs/{RUN_NAME}/step_{s}_ft_final"
#     if os.path.exists(stale_train_dir):
#         shutil.rmtree(stale_train_dir)
#     if os.path.exists(stale_final_dir):
#         shutil.rmtree(stale_final_dir)

# first_step_classes = classes_for_step(0)

# def _label_range(ds, n=200):
#     vals = [int(ds[i]["label"]) for i in range(min(n, len(ds)))]
#     return min(vals), max(vals)

# base_model_dir = STEP1_FINAL_OUT

# for step_idx in range(1, num_steps):
#     train_step, test_step, label2new, new2orig, class_ids = make_step_datasets(
#         step_idx,
#         split_type="new_only",
#         remap_labels=False,
#     )

#     print(f"\n[FT] Step {step_idx+1}")
#     print("Current step classes:", class_ids[:5], "...", class_ids[-5:])

#     tr_min, tr_max = _label_range(train_step)
#     te_min, te_max = _label_range(test_step)

#     print("Train label range:", tr_min, tr_max)
#     print("Test label range:", te_min, te_max)
#     print("Num labels:", num_classes)

#     print("Loaded base_model_dir:", base_model_dir)
#     ft_model = AutoModelForImageClassification.from_pretrained(base_model_dir)

#     print("Loaded FT model num_labels:", ft_model.config.num_labels)
#     print("Loaded FT classifier out_features:", ft_model.classifier.out_features)

#     assert ft_model.config.num_labels == num_classes, (
#         f"Loaded FT model num_labels={ft_model.config.num_labels}, expected {num_classes}"
#     )
#     assert ft_model.classifier.out_features == num_classes, (
#         f"Loaded FT classifier out_features={ft_model.classifier.out_features}, expected {num_classes}"
#     )

#     assert tr_min >= 0
#     assert tr_max < ft_model.classifier.out_features, (
#         f"Train labels out of range: max label {tr_max}, classifier out_features {ft_model.classifier.out_features}"
#     )
#     assert te_min >= 0
#     assert te_max < ft_model.classifier.out_features, (
#         f"Test labels out of range: max label {te_max}, classifier out_features {ft_model.classifier.out_features}"
#     )

#     step_ft_train_dir = f"outputs/{RUN_NAME}/step_{step_idx+1}_ft"

#     args_ft = TrainingArguments(
#         output_dir=step_ft_train_dir,
#         remove_unused_columns=False,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         save_total_limit=2,
#         num_train_epochs=FT_EPOCHS,
#         learning_rate=LR_FT,
#         weight_decay=0.01,
#         warmup_ratio=WARMUP_RATIO,
#         lr_scheduler_type=SCHED,
#         per_device_train_batch_size=BATCH_FT,
#         per_device_eval_batch_size=32,
#         gradient_accumulation_steps=ACCUM_FT,
#         fp16=USE_FP16,
#         dataloader_num_workers=4,
#         logging_steps=50,
#         load_best_model_at_end=True,
#         metric_for_best_model="accuracy",
#         greater_is_better=True,
#         report_to="none",
#         max_grad_norm=1.0,
#     )

#     trainer_ft = Trainer(
#         model=ft_model,
#         args=args_ft,
#         train_dataset=train_step,
#         eval_dataset=test_step,
#         data_collator=collate_fn,
#         compute_metrics=compute_metrics,
#     )

#     trainer_ft.train()
#     eval_current = trainer_ft.evaluate(test_step)

#     pd.DataFrame(trainer_ft.state.log_history).to_csv(
#         os.path.join(TABLES_DIR, f"step{step_idx+1}_ft_log_history.csv"),
#         index=False
#     )

#     step_ft_dir = f"outputs/{RUN_NAME}/step_{step_idx+1}_ft_final"
#     os.makedirs(step_ft_dir, exist_ok=True)

#     print(f"[FT] Step {step_idx+1} saving model to {step_ft_dir}")
#     trainer_ft.save_model(step_ft_dir)
#     image_processor.save_pretrained(step_ft_dir)
#     print(f"[FT] Step {step_idx+1} save finished")

#     reloaded_ft = AutoModelForImageClassification.from_pretrained(step_ft_dir)
#     print(f"[FT] Reload check step {step_idx+1} num_labels:", reloaded_ft.config.num_labels)
#     print(f"[FT] Reload check step {step_idx+1} classifier out_features:", reloaded_ft.classifier.out_features)

#     assert reloaded_ft.config.num_labels == num_classes, (
#         f"Reloaded saved FT num_labels={reloaded_ft.config.num_labels}, expected {num_classes}"
#     )
#     assert reloaded_ft.classifier.out_features == num_classes, (
#         f"Reloaded saved FT classifier out_features={reloaded_ft.classifier.out_features}, expected {num_classes}"
#     )

#     base_model_dir = step_ft_dir

#     seen_classes_now = classes_for_cumulative(step_idx)
#     eval_first = make_eval_dataset(first_step_classes, name=f"ft_step{step_idx+1}_first_step")

#     later_seen_now = [c for c in seen_classes_now if c not in first_step_classes]
#     eval_later = make_eval_dataset(later_seen_now, name=f"ft_step{step_idx+1}_later_seen") if len(later_seen_now) > 0 else None
#     eval_seen = make_eval_dataset(seen_classes_now, name=f"ft_step{step_idx+1}_all_seen")

#     metrics_first = trainer_ft.evaluate(eval_first)
#     metrics_seen = trainer_ft.evaluate(eval_seen)

#     if eval_later is not None:
#         metrics_later = trainer_ft.evaluate(eval_later)
#     else:
#         metrics_later = {"eval_accuracy": np.nan, "eval_loss": np.nan}

#     ft_results.extend([
#         {
#             "experiment": f"ft_step_{step_idx+1}",
#             "method": "full_finetune",
#             "step": step_idx + 1,
#             "eval_type": "current_step",
#             "eval_accuracy": float(eval_current.get("eval_accuracy", np.nan)),
#             "eval_loss": float(eval_current.get("eval_loss", np.nan)),
#         },
#         {
#             "experiment": f"ft_step_{step_idx+1}",
#             "method": "full_finetune",
#             "step": step_idx + 1,
#             "eval_type": "first_step",
#             "eval_accuracy": float(metrics_first.get("eval_accuracy", np.nan)),
#             "eval_loss": float(metrics_first.get("eval_loss", np.nan)),
#         },
#         {
#             "experiment": f"ft_step_{step_idx+1}",
#             "method": "full_finetune",
#             "step": step_idx + 1,
#             "eval_type": "later_steps_seen_so_far",
#             "eval_accuracy": float(metrics_later.get("eval_accuracy", np.nan)),
#             "eval_loss": float(metrics_later.get("eval_loss", np.nan)),
#         },
#         {
#             "experiment": f"ft_step_{step_idx+1}",
#             "method": "full_finetune",
#             "step": step_idx + 1,
#             "eval_type": "all_seen",
#             "eval_accuracy": float(metrics_seen.get("eval_accuracy", np.nan)),
#             "eval_loss": float(metrics_seen.get("eval_loss", np.nan)),
#         },
#     ])

# print("\nFull FT continual training finished.")


# In[ ]:


# final_ft_dir = f"outputs/{RUN_NAME}/step_{num_steps}_ft_final"
# final_ft_model = AutoModelForImageClassification.from_pretrained(final_ft_dir)

# args_ft_eval = TrainingArguments(
#     output_dir=f"outputs/{RUN_NAME}/final_ft_eval",
#     remove_unused_columns=False,
#     report_to="none",
#     fp16=USE_FP16,
#     per_device_eval_batch_size=32,
# )

# trainer_ft_final = Trainer(
#     model=final_ft_model,
#     args=args_ft_eval,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
# )

# ft_test_first = make_eval_dataset(first_step_classes)
# ft_test_later = make_eval_dataset(later_step_classes)
# ft_test_seen = make_eval_dataset(final_seen_classes)

# print("Final FT num_labels:", final_ft_model.config.num_labels)
# print("Final FT classifier out_features:", final_ft_model.classifier.out_features)
# assert final_ft_model.classifier.out_features == num_classes

# ft_final_first = trainer_ft_final.evaluate(ft_test_first)
# ft_final_later = trainer_ft_final.evaluate(ft_test_later)
# ft_final_seen = trainer_ft_final.evaluate(ft_test_seen)

# print("FT final - first step:", ft_final_first)
# print("FT final - later steps:", ft_final_later)
# print("FT final - all seen:", ft_final_seen)


# ## 10) Upper bound: joint training (full dataset)

# In[ ]:


# train_joint, test_joint, label2new_J, new2orig_J, class_ids_J = make_step_datasets(
#     step_idx=0, split_type="full", remap_labels=False
# )

# config_joint = AutoConfig.from_pretrained(
#     model_checkpoint,
#     num_labels=num_classes,
#     id2label={i: str(i) for i in range(num_classes)},
#     label2id={str(i): i for i in range(num_classes)},
# )

# joint_model = AutoModelForImageClassification.from_config(config_joint)

# args_joint = TrainingArguments(
#     output_dir=JOINT_OUT,
#     remove_unused_columns=False,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     num_train_epochs=JOINT_EPOCHS,
#     learning_rate=LR_JOINT,
#     weight_decay=WEIGHT_DECAY,
#     warmup_ratio=WARMUP_RATIO,
#     lr_scheduler_type=SCHED,
#     per_device_train_batch_size=BATCH_FT,
#     per_device_eval_batch_size=32,
#     gradient_accumulation_steps=ACCUM_FT,
#     fp16=USE_FP16,
#     dataloader_num_workers=4,
#     logging_steps=50,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     greater_is_better=True,
#     report_to="none",
#     max_grad_norm=1.0,
# )

# trainer_joint = Trainer(
#     model=joint_model,
#     args=args_joint,
#     train_dataset=train_joint,
#     eval_dataset=test_joint,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
# )

# trainer_joint.train()

# eval_joint = trainer_joint.evaluate()
# print({"eval_joint_full": eval_joint})

# first_step_classes = classes_for_step(0)

# later_step_classes = []
# for s in range(1, num_steps):
#     later_step_classes.extend(classes_for_step(s))

# final_seen_classes = classes_for_cumulative(num_steps - 1)

# joint_eval_first = make_eval_dataset(first_step_classes, name="joint_first")
# joint_eval_later = make_eval_dataset(later_step_classes, name="joint_later")
# joint_eval_seen  = make_eval_dataset(final_seen_classes, name="joint_seen")

# joint_final_first = trainer_joint.evaluate(eval_dataset=joint_eval_first)
# joint_final_later = trainer_joint.evaluate(eval_dataset=joint_eval_later)
# joint_final_seen  = trainer_joint.evaluate(eval_dataset=joint_eval_seen)

# print("Joint final - first_step:", joint_final_first)
# print("Joint final - later_steps:", joint_final_later)
# print("Joint final - all_seen:", joint_final_seen)


# In[ ]:


# joint_log_df = pd.DataFrame(trainer_joint.state.log_history)
# joint_log_df.to_csv(os.path.join(TABLES_DIR, "joint_log_history.csv"), index=False)

# joint_metrics = {
#     "experiment": "joint_full",
#     "eval_loss": float(eval_joint.get("eval_loss", np.nan)),
#     "eval_accuracy": float(eval_joint.get("eval_accuracy", np.nan)),
# }

# with open(os.path.join(METRICS_DIR, "joint_metrics.json"), "w") as f:
#     json.dump(joint_metrics, f, indent=2)

# joint_log_df.tail()


# In[ ]:


# joint_test_first = make_eval_dataset(first_step_classes)
# joint_test_later = make_eval_dataset(later_step_classes)
# joint_test_all = make_eval_dataset(all_classes)

# joint_final_first = trainer_joint.evaluate(joint_test_first)
# joint_final_later = trainer_joint.evaluate(joint_test_later)
# joint_final_all = trainer_joint.evaluate(joint_test_all)

# print("Joint final - first step:", joint_final_first)
# print("Joint final - later steps:", joint_final_later)
# print("Joint final - all classes:", joint_final_all)


# ## 11) Compare results (step test vs full test)

# In[ ]:


# def grab_acc(d):
#     return float(d["eval_accuracy"]) if "eval_accuracy" in d else np.nan

# def grab_loss(d):
#     return float(d["eval_loss"]) if "eval_loss" in d else np.nan

# all_results = []

# all_results.extend(results)
# all_results.extend(lora_results)
# all_results.extend(ft_results)

# # final simple avg variants
# for variant_name, eval_dict in merged_final_evals.items():
#     all_results.extend([
#         {
#             "experiment": f"{variant_name}_final_eval",
#             "method": f"simple_avg_{variant_name}",
#             "step": num_steps,
#             "eval_type": "first_step",
#             "eval_accuracy": grab_acc(eval_dict["first_step"]),
#             "eval_loss": grab_loss(eval_dict["first_step"]),
#         },
#         {
#             "experiment": f"{variant_name}_final_eval",
#             "method": f"simple_avg_{variant_name}",
#             "step": num_steps,
#             "eval_type": "later_steps",
#             "eval_accuracy": grab_acc(eval_dict["later_steps"]),
#             "eval_loss": grab_loss(eval_dict["later_steps"]),
#         },
#         {
#             "experiment": f"{variant_name}_final_eval",
#             "method": f"simple_avg_{variant_name}",
#             "step": num_steps,
#             "eval_type": "all_seen",
#             "eval_accuracy": grab_acc(eval_dict["all_seen"]),
#             "eval_loss": grab_loss(eval_dict["all_seen"]),
#         },
#     ])

# # FT
# all_results.extend([
#     {"experiment": "ft_final_eval", "method": "full_finetune", "step": num_steps, "eval_type": "first_step", "eval_accuracy": grab_acc(ft_final_first), "eval_loss": grab_loss(ft_final_first)},
#     {"experiment": "ft_final_eval", "method": "full_finetune", "step": num_steps, "eval_type": "later_steps", "eval_accuracy": grab_acc(ft_final_later), "eval_loss": grab_loss(ft_final_later)},
#     {"experiment": "ft_final_eval", "method": "full_finetune", "step": num_steps, "eval_type": "all_seen", "eval_accuracy": grab_acc(ft_final_seen), "eval_loss": grab_loss(ft_final_seen)},
# ])

# # JOINT
# all_results.extend([
#     {"experiment": "joint_final_eval", "method": "joint_upper_bound", "step": num_steps, "eval_type": "first_step", "eval_accuracy": grab_acc(joint_final_first), "eval_loss": grab_loss(joint_final_first)},
#     {"experiment": "joint_final_eval", "method": "joint_upper_bound", "step": num_steps, "eval_type": "later_steps", "eval_accuracy": grab_acc(joint_final_later), "eval_loss": grab_loss(joint_final_later)},
#     {"experiment": "joint_final_eval", "method": "joint_upper_bound", "step": num_steps, "eval_type": "all_seen", "eval_accuracy": grab_acc(joint_final_seen), "eval_loss": grab_loss(joint_final_seen)},
# ])

# results_df = pd.DataFrame(all_results)
# print(results_df)


# In[ ]:


# results_df_clean = results_df.copy()

# results_df_clean = results_df_clean.rename(columns={
#     "eval_accuracy": "accuracy",
#     "eval_loss": "loss",
#     "eval_type": "eval_set"
# })

# def map_method(x):
#     if x == "step1_scratch":
#         return "B0"
#     elif x == "lora_normal":
#         return "LoRA Normal"
#     elif x == "lora_orth":
#         return "LoRA Orth"
#     elif x == "lora_replay":
#         return "LoRA Replay"
#     elif x == "simple_avg_normal":
#         return "Simple AVG - Normal"
#     elif x == "simple_avg_orth":
#         return "Simple AVG - Orth"
#     elif x == "simple_avg_replay":
#         return "Simple AVG - Replay"
#     elif x == "full_finetune":
#         return "Full FT"
#     elif x == "joint_upper_bound":
#         return "Joint"
#     return x

# results_df_clean["method"] = results_df_clean["method"].apply(map_method)

# def map_eval_stage(exp_name):
#     if "final_eval" in exp_name:
#         return "final"
#     elif exp_name == "step1_scratch":
#         return "step1"
#     return "step"

# results_df_clean["eval_stage"] = results_df_clean["experiment"].apply(map_eval_stage)

# def map_adapter_name(row):
#     if row["method"] == "B0":
#         return "B0"

#     if row["method"] in ["LoRA Normal", "LoRA Orth", "LoRA Replay"]:
#         step = int(row["step"])
#         if step == 2:
#             return "L1"
#         elif step == 3:
#             return "L2"
#         elif step == 4:
#             return "L3"
#         elif step == 5:
#             return "L4"

#     if row["method"] == "Simple AVG - Normal":
#         return "L1234_avg_normal"
#     if row["method"] == "Simple AVG - Orth":
#         return "L1234_avg_orth"
#     if row["method"] == "Simple AVG - Replay":
#         return "L1234_avg_replay"

#     return np.nan

# results_df_clean["adapter_name"] = results_df_clean.apply(map_adapter_name, axis=1)

# results_df_clean = results_df_clean[
#     ["method", "adapter_name", "step", "eval_stage", "eval_set", "accuracy", "loss"]
# ].copy()

# results_df_clean.to_csv(os.path.join(TABLES_DIR, "results_summary_clean.csv"), index=False)

# print(results_df_clean)


# In[ ]:


# summary_lines = [
#     "Experiment summary",
#     "==================",
#     "",
# ]

# for _, row in results_df_clean.iterrows():
#     acc = row["accuracy"]
#     loss = row["loss"]

#     acc_str = f"{acc:.4f}" if pd.notna(acc) else "nan"
#     loss_str = f"{loss:.4f}" if pd.notna(loss) else "nan"

#     summary_lines.append(
#         f"method={row['method']} | adapter_name={row['adapter_name']} | "
#         f"step={row['step']} | eval_stage={row['eval_stage']} | "
#         f"eval_set={row['eval_set']} | accuracy={acc_str} | loss={loss_str}"
#     )

# with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
#     f.write("\n".join(summary_lines))

# print("\n".join(summary_lines))


# In[ ]:


# import matplotlib.pyplot as plt
# import numpy as np

# df_final = results_df_clean[
#     (results_df_clean["eval_stage"] == "final") &
#     (results_df_clean["step"] == num_steps)
# ].copy()

# df_final = df_final[df_final["method"].isin([
#     "Simple AVG - Normal",
#     "Simple AVG - Orth",
#     "Simple AVG - Replay",
#     "Full FT",
#     "Joint"
# ])]

# labels = ["first_step", "later_steps", "all_seen"]

# pivot = df_final.pivot_table(
#     index="eval_set",
#     columns="method",
#     values="accuracy",
#     aggfunc="mean"
# ).reindex(labels)

# method_order = [
#     "Simple AVG - Normal",
#     "Simple AVG - Orth",
#     "Simple AVG - Replay",
#     "Full FT",
#     "Joint",
# ]
# pivot = pivot[method_order]

# x = np.arange(len(labels))
# width = 0.16

# plt.figure(figsize=(13, 5))
# for i, method_name in enumerate(method_order):
#     plt.bar(x + (i - 2) * width, pivot[method_name], width, label=method_name)

# plt.xticks(x, labels)
# plt.ylabel("Accuracy")
# plt.title("Final Accuracy Comparison")
# plt.legend()
# plt.tight_layout()

# plot_path = os.path.join(PLOTS_DIR, "final_accuracy_comparison.png")
# plt.savefig(plot_path, dpi=200, bbox_inches="tight")
# plt.show()

# print("Saved plot to:", plot_path)


# In[ ]:


# final_first = df_final[df_final["eval_set"] == "first_step"]
# final_later = df_final[df_final["eval_set"] == "later_steps"]
# final_seen  = df_final[df_final["eval_set"] == "all_seen"]

# def make_barplot(df_sub, title, filename):
#     methods = ["Merged LoRA", "Full FT", "Joint"]
#     vals = []
#     for m in methods:
#         row = df_sub[df_sub["method"] == m]
#         vals.append(float(row["accuracy"].iloc[0]) if len(row) > 0 else np.nan)

#     plt.figure(figsize=(7, 4))
#     plt.bar(methods, vals)
#     plt.ylabel("Accuracy")
#     plt.title(title)
#     plt.tight_layout()

#     out_path = os.path.join(PLOTS_DIR, filename)
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.show()
#     print("Saved plot to:", out_path)

# make_barplot(final_first, "Final Accuracy on B0 Classes", "final_first_step_accuracy_lora_ft_joint.png")
# make_barplot(final_later, "Final Accuracy on Later Classes", "final_later_steps_accuracy_lora_ft_joint.png")
# make_barplot(final_seen,  "Final Accuracy on All Seen Classes", "final_all_seen_accuracy_lora_ft_joint.png")


# In[ ]:


# plot_df = results_df_clean.copy()
# plot_df_step = plot_df[plot_df["eval_stage"] == "step"].copy()

# lora_current = plot_df_step[
#     (plot_df_step["method"] == "LoRA separate") &
#     (plot_df_step["eval_set"] == "current_step")
# ].copy()

# plt.figure(figsize=(7, 4))
# plt.plot(lora_current["step"], lora_current["accuracy"], marker="o")
# plt.xticks(lora_current["step"], lora_current["adapter_name"])
# plt.xlabel("Adapter")
# plt.ylabel("Accuracy")
# plt.title("LoRA Separate: Current-Step Accuracy over L1..L4")
# plt.tight_layout()

# out = os.path.join(PLOTS_DIR, "curve_lora_separate_current_step.png")
# plt.savefig(out, dpi=200, bbox_inches="tight")
# plt.show()
# print("Saved:", out)


# In[ ]:


# def make_barplot(df_sub, title, filename):
#     methods = [
#         "Simple AVG - Normal",
#         "Simple AVG - Orth",
#         "Simple AVG - Replay",
#         "Full FT",
#         "Joint",
#     ]
#     vals = []
#     for m in methods:
#         row = df_sub[df_sub["method"] == m]
#         vals.append(float(row["accuracy"].iloc[0]) if len(row) > 0 else np.nan)

#     plt.figure(figsize=(10, 4))
#     plt.bar(methods, vals)
#     plt.ylabel("Accuracy")
#     plt.title(title)
#     plt.xticks(rotation=20)
#     plt.tight_layout()

#     out_path = os.path.join(PLOTS_DIR, filename)
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.show()
#     print("Saved plot to:", out_path)

# final_first = df_final[df_final["eval_set"] == "first_step"]
# final_later = df_final[df_final["eval_set"] == "later_steps"]
# final_seen  = df_final[df_final["eval_set"] == "all_seen"]

# make_barplot(final_first, "Final Accuracy on B0 Classes", "final_first_step_all_methods.png")
# make_barplot(final_later, "Final Accuracy on Later Classes", "final_later_step_all_methods.png")
# make_barplot(final_seen, "Final Accuracy on All Seen Classes", "final_all_seen_all_methods.png")


# In[ ]:


# df_merge_only = df_final[df_final["method"].isin([
#     "Simple AVG - Normal",
#     "Simple AVG - Orth",
#     "Simple AVG - Replay"
# ])].copy()

# pivot_merge = df_merge_only.pivot_table(
#     index="eval_set",
#     columns="method",
#     values="accuracy",
#     aggfunc="mean"
# ).reindex(["first_step", "later_steps", "all_seen"])

# method_order_merge = [
#     "Simple AVG - Normal",
#     "Simple AVG - Orth",
#     "Simple AVG - Replay"
# ]
# pivot_merge = pivot_merge[method_order_merge]

# x = np.arange(3)
# width = 0.22

# plt.figure(figsize=(10, 5))
# for i, method_name in enumerate(method_order_merge):
#     plt.bar(x + (i - 1) * width, pivot_merge[method_name], width, label=method_name)

# plt.xticks(x, ["first_step", "later_steps", "all_seen"])
# plt.ylabel("Accuracy")
# plt.title("Effect of Variant under Simple AVG Merge")
# plt.legend()
# plt.tight_layout()

# plot_path = os.path.join(PLOTS_DIR, "simple_avg_variant_comparison.png")
# plt.savefig(plot_path, dpi=200, bbox_inches="tight")
# plt.show()

# print("Saved plot to:", plot_path)


# In[ ]:


# merge_summary = df_final[df_final["method"].isin([
#     "Simple AVG - Normal",
#     "Simple AVG - Orth",
#     "Simple AVG - Replay"
# ])].copy()

# print(merge_summary.sort_values(["eval_set", "method"]))

