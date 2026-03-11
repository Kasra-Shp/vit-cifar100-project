#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Set these BEFORE loading datasets/models (edit paths if you prefer a different NFS folder)
os.environ.setdefault("HF_HOME", "/nfsd/lttm4/tesisti/shahrampour/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/nfsd/lttm4/tesisti/shahrampour/hf_datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", "/nfsd/lttm4/tesisti/shahrampour/hf_transformers")

for k in ["HF_HOME","HF_DATASETS_CACHE","TRANSFORMERS_CACHE"]:
    os.makedirs(os.environ[k], exist_ok=True)

print("HF_HOME:", os.environ["HF_HOME"])
print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
print("TRANSFORMERS_CACHE:", os.environ["TRANSFORMERS_CACHE"])


# ## 1) Imports

# In[2]:


import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt


from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from peft import LoraConfig, get_peft_model, TaskType

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


# In[ ]:


RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
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


# ## 4) Model + preprocessing

# In[ ]:


# Requested model
model_checkpoint = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)

from torchvision import transforms

# Processor size could be int or dict (height/width)
size = image_processor.size
if isinstance(size, dict):
    H = size.get("height", 224)
    W = size.get("width", 224)
else:
    H = W = int(size)

train_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.RandomCrop((H, W), padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

val_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])
from PIL import Image
import numpy as np

from PIL import Image
import numpy as np

def to_pil(x):
    # already PIL
    if isinstance(x, Image.Image):
        return x

    # HF image dict
    if isinstance(x, dict):
        if "array" in x:
            x = x["array"]
        elif "bytes" in x:
            import io
            return Image.open(io.BytesIO(x["bytes"])).convert("RGB")

    # list -> np
    if isinstance(x, list):
        x = np.array(x, dtype=np.uint8)

    # numpy -> fix shape
    if isinstance(x, np.ndarray):
        arr = x.astype(np.uint8)

        # remove trivial dims, e.g. (1,1,32,3) -> (32,3)?? or (1,32,32,3)->(32,32,3)
        arr = np.squeeze(arr)

        # if grayscale (H,W) -> make RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        # if channel-first (C,H,W) -> (H,W,C)
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[-1] not in (1,3):
            arr = np.transpose(arr, (1,2,0))

        # if still not 3D HWC, raise a helpful error
        if not (arr.ndim == 3 and arr.shape[-1] in (1,3)):
            raise TypeError(f"Unexpected image array shape after squeeze: {arr.shape}")

        # ensure 3 channels
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return Image.fromarray(arr)

    # fallback
    return x
    
def preprocess_train(ex):
    ex["pixel_values"] = [train_transform(to_pil(img)) for img in ex["img"]]
    return ex

def preprocess_val(ex):
    ex["pixel_values"] = [val_transform(to_pil(img)) for img in ex["img"]]
    return ex
    
def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples], dtype=torch.long)
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

    ds = ds.with_transform(None)
    ds.reset_format()

    return ds.filter(lambda x: int(x["label"]) in class_set)

def make_step_datasets(step_idx: int, split_type: str = "step", remap_labels: bool = True):
    """
    split_type:
      - 'step'       : only classes of this step
      - 'cumulative' : union of classes up to this step
      - 'full'       : all classes (100)
    """
    if split_type == "full":
        class_ids = list(range(num_classes))
    elif split_type == "cumulative":
        class_ids = classes_for_cumulative(step_idx)
    else:
        class_ids = classes_for_step(step_idx)

    # IMPORTANT: filter first, set transforms later
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
        label2new = None
        new2orig = None

    train_ds.set_transform(preprocess_train)
    test_ds.set_transform(preprocess_val)

    return train_ds, test_ds, label2new, new2orig, class_ids

# keep full test set only for 100-class evaluation
eval_all = dataset["test"]
eval_all.reset_format()
eval_all.set_transform(preprocess_val)

print("eval_all size:", len(eval_all))


# In[ ]:


run_config = {
    "model_checkpoint": model_checkpoint,
    "num_classes": num_classes,
    "num_steps": num_steps,
    "classes_per_step": classes_per_step,
}

with open(os.path.join(METRICS_DIR, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)


# ## 6) Training recipes (reasonable settings)

# In[ ]:


set_seed(42)

SCRATCH_EPOCHS = 15
LORA_EPOCHS    = 10
FT_EPOCHS      = 10
JOINT_EPOCHS   = 15

# SCRATCH_EPOCHS = 1
# LORA_EPOCHS    = 1
# FT_EPOCHS      = 1
# JOINT_EPOCHS   = 1

BATCH_SCRATCH = 8
ACCUM_SCRATCH = 2

BATCH_LORA    = 16
ACCUM_LORA    = 1

BATCH_FT      = 8
ACCUM_FT      = 2

LR_SCRATCH = 5e-5
LR_LORA    = 5e-4
LR_FT      = 3e-5
LR_JOINT   = 5e-5

WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.10
SCHED = "cosine"

USE_FP16 = torch.cuda.is_available()


# ## 7) Step 1: train full ViT from scratch on step 0 classes

# In[ ]:


step1_idx = 0
train_step1, test_step1, label2new_1, new2orig_1, class_ids_1 = make_step_datasets(
    step1_idx, split_type="step", remap_labels=True
)
num_labels_1 = len(label2new_1)

print("Step1 original classes:", class_ids_1[:5], "...", class_ids_1[-5:])
print("Step1 num_labels (head):", num_labels_1)

model_step1 = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels_1,
    id2label={i: str(new2orig_1[i]) for i in range(num_labels_1)},
    label2id={str(new2orig_1[i]): i for i in range(num_labels_1)},
    ignore_mismatched_sizes=True,
)

args_step1 = TrainingArguments(
    output_dir="outputs/step1_pretrained",
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

SAVE_STEP1_DIR = "outputs/step1_pretrained_final"
trainer_step1.save_model(SAVE_STEP1_DIR)
image_processor.save_pretrained(SAVE_STEP1_DIR)


# In[ ]:


step1_log_df = pd.DataFrame(trainer_step1.state.log_history)
step1_log_df.to_csv(os.path.join(TABLES_DIR, "step1_log_history.csv"), index=False)

step1_metrics = {
    "experiment": "step1_pretrained",
    "eval_loss": float(eval_step1.get("eval_loss", np.nan)),
    "eval_accuracy": float(eval_step1.get("eval_accuracy", np.nan)),
}

with open(os.path.join(METRICS_DIR, "step1_metrics.json"), "w") as f:
    json.dump(step1_metrics, f, indent=2)

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


# ## 8) Step 2: LoRA only (freeze backbone) on top of Step 1 model

# In[ ]:


step2_idx = 1
train_step2, test_step2, label2new_2, new2orig_2, class_ids_2 = make_step_datasets(
    step2_idx, split_type="step", remap_labels=True
)
num_labels_2 = len(label2new_2)

print("Step2 original classes:", class_ids_2[:5], "...", class_ids_2[-5:])
print("Step2 num_labels (head):", num_labels_2)

base_model_dir = "outputs/step1_pretrained_final"

base_model = AutoModelForImageClassification.from_pretrained(
    base_model_dir,
    num_labels=num_labels_2,
    id2label={i: str(new2orig_2[i]) for i in range(num_labels_2)},
    label2id={str(new2orig_2[i]): i for i in range(num_labels_2)},
    ignore_mismatched_sizes=True,
)

for p in base_model.parameters():
    p.requires_grad = False

for n, p in base_model.named_parameters():
    if "classifier" in n:
        p.requires_grad = True

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)

model_lora_step2 = get_peft_model(base_model, lora_config)

trainable = sum(p.numel() for p in model_lora_step2.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_lora_step2.parameters())
print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")

args_lora = TrainingArguments(
    output_dir="outputs/step2_lora",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=LORA_EPOCHS,
    learning_rate=LR_LORA,
    weight_decay=0.01,
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

trainer_lora = Trainer(
    model=model_lora_step2,
    args=args_lora,
    train_dataset=train_step2,
    eval_dataset=test_step2,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer_lora.train()

eval_step2_lora = trainer_lora.evaluate()
print({"eval_step2_lora": eval_step2_lora})


# In[ ]:


step2_lora_log_df = pd.DataFrame(trainer_lora.state.log_history)
step2_lora_log_df.to_csv(os.path.join(TABLES_DIR, "step2_lora_log_history.csv"), index=False)

step2_lora_metrics = {
    "experiment": "step2_lora",
    "eval_loss": float(eval_step2_lora.get("eval_loss", np.nan)),
    "eval_accuracy": float(eval_step2_lora.get("eval_accuracy", np.nan)),
}

with open(os.path.join(METRICS_DIR, "step2_lora_metrics.json"), "w") as f:
    json.dump(step2_lora_metrics, f, indent=2)

step2_lora_log_df.tail()


# ## 9) Baseline: full fine-tune instead of LoRA (Step 2)

# In[ ]:


ft_model = AutoModelForImageClassification.from_pretrained(
    base_model_dir,
    num_labels=num_labels_2,
    id2label={i: str(new2orig_2[i]) for i in range(num_labels_2)},
    label2id={str(new2orig_2[i]): i for i in range(num_labels_2)},
    ignore_mismatched_sizes=True,
)

for p in ft_model.parameters():
    p.requires_grad = True

args_ft = TrainingArguments(
    output_dir="outputs/step2_finetune_full",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=FT_EPOCHS,
    learning_rate=LR_FT,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=SCHED,
    per_device_train_batch_size=BATCH_FT,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=ACCUM_FT,
    fp16=USE_FP16,
    dataloader_num_workers=4,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    max_grad_norm=1.0,
)

trainer_ft = Trainer(
    model=ft_model,
    args=args_ft,
    train_dataset=train_step2,
    eval_dataset=test_step2,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer_ft.train()

eval_step2_ft = trainer_ft.evaluate()
print({"eval_step2_ft": eval_step2_ft})


# In[ ]:


step2_ft_log_df = pd.DataFrame(trainer_ft.state.log_history)
step2_ft_log_df.to_csv(os.path.join(TABLES_DIR, "step2_ft_log_history.csv"), index=False)

step2_ft_metrics = {
    "experiment": "step2_full_finetune",
    "eval_loss": float(eval_step2_ft.get("eval_loss", np.nan)),
    "eval_accuracy": float(eval_step2_ft.get("eval_accuracy", np.nan)),
}

with open(os.path.join(METRICS_DIR, "step2_ft_metrics.json"), "w") as f:
    json.dump(step2_ft_metrics, f, indent=2)

step2_ft_log_df.tail()


# ## 10) Upper bound: joint training (full dataset)

# In[ ]:


train_joint, test_joint, label2new_J, new2orig_J, class_ids_J = make_step_datasets(
    step_idx=0, split_type="full", remap_labels=True
)
num_labels_J = len(label2new_J)

joint_model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels_J,
    id2label={i: str(new2orig_J[i]) for i in range(num_labels_J)},
    label2id={str(new2orig_J[i]): i for i in range(num_labels_J)},
    ignore_mismatched_sizes=True,
)

args_joint = TrainingArguments(
    output_dir="outputs/joint_full",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=JOINT_EPOCHS,
    learning_rate=LR_JOINT,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=SCHED,
    per_device_train_batch_size=BATCH_FT,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=ACCUM_FT,
    fp16=USE_FP16,
    dataloader_num_workers=4,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    max_grad_norm=1.0,
)

trainer_joint = Trainer(
    model=joint_model,
    args=args_joint,
    train_dataset=train_joint,
    eval_dataset=test_joint,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer_joint.train()

eval_joint = trainer_joint.evaluate()
print({"eval_joint_full": eval_joint})


# In[ ]:


joint_log_df = pd.DataFrame(trainer_joint.state.log_history)
joint_log_df.to_csv(os.path.join(TABLES_DIR, "joint_log_history.csv"), index=False)

joint_metrics = {
    "experiment": "joint_full",
    "eval_loss": float(eval_joint.get("eval_loss", np.nan)),
    "eval_accuracy": float(eval_joint.get("eval_accuracy", np.nan)),
}

with open(os.path.join(METRICS_DIR, "joint_metrics.json"), "w") as f:
    json.dump(joint_metrics, f, indent=2)

joint_log_df.tail()


# ## 11) Compare results (step test vs full test)

# In[ ]:


def grab_acc(d):
    return float(d["eval_accuracy"]) if "eval_accuracy" in d else np.nan

def grab_loss(d):
    return float(d["eval_loss"]) if "eval_loss" in d else np.nan

results_rows = [
    {
        "experiment": "step1_pretrained",
        "method": "full_finetune",
        "step": 1,
        "eval_accuracy": grab_acc(eval_step1),
        "eval_loss": grab_loss(eval_step1),
    },
    {
        "experiment": "step2_lora",
        "method": "lora",
        "step": 2,
        "eval_accuracy": grab_acc(eval_step2_lora),
        "eval_loss": grab_loss(eval_step2_lora),
    },
    {
        "experiment": "step2_full_finetune",
        "method": "full_finetune",
        "step": 2,
        "eval_accuracy": grab_acc(eval_step2_ft),
        "eval_loss": grab_loss(eval_step2_ft),
    },
    {
        "experiment": "joint_full",
        "method": "joint_upper_bound",
        "step": 0,
        "eval_accuracy": grab_acc(eval_joint),
        "eval_loss": grab_loss(eval_joint),
    },
]

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(TABLES_DIR, "results_summary.csv"), index=False)

results_df


# In[ ]:


plt.figure(figsize=(8,5))
plt.bar(results_df["experiment"], results_df["eval_accuracy"])
plt.ylabel("Eval Accuracy")
plt.title("Accuracy Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"), dpi=200)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(results_df["experiment"], results_df["eval_loss"])
plt.ylabel("Eval Loss")
plt.title("Loss Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "loss_comparison.png"), dpi=200)
plt.show()

compare_df = results_df[results_df["experiment"].isin(["step2_lora", "step2_full_finetune"])]

plt.figure(figsize=(6,4))
plt.bar(compare_df["experiment"], compare_df["eval_accuracy"])
plt.ylabel("Eval Accuracy")
plt.title("LoRA vs Full Fine-Tuning")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "lora_vs_ft_accuracy.png"), dpi=200)
plt.show()


# In[ ]:


summary_lines = [
    "Experiment summary",
    "==================",
    "",
]

for _, row in results_df.iterrows():
    summary_lines.append(
        f"{row['experiment']} | method={row['method']} | step={row['step']} | "
        f"eval_accuracy={row['eval_accuracy']:.4f} | eval_loss={row['eval_loss']:.4f}"
    )

with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))

