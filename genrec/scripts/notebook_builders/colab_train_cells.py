# %% [markdown]
# # Fine-tune Qwen2.5-0.5B as a movie recommender (Google Colab)
#
# Trains the generative recommender with the `genrec` package (installed from GitHub below). Every sample is a chat
# turn; the loss covers the assistant answer only.
#
# | task | question | answer |
# |---|---|---|
# | `plot` | What is the movie "*title*" about? | Wikipedia plot (first `plot_words` words) |
# | `genres` | Which genres does "*title*" belong to? | genres |
# | `details` | Who made "*title*" and who stars in it? | director + main cast |
# | `identify` | Which movie is this? *plot snippet* | title (year) |
# | `next_movie` | A user rated these movies, oldest first ... which next? | title (year) |
#
# **Recipes** (ML-1M, test HR@10 on the same 1,000 users):
#
# | run | start from | `next_per_user` | epochs | HR@10 |
# |---|---|---|---|---|
# | v1 | `Qwen/Qwen2.5-0.5B` | 5 | 4 | 0.160 |
# | v2 | the v1 model | 20 | 2 (the best checkpoint; epochs 3-4 overfit) | 0.236 |
#
# **Before you run:** Runtime → Change runtime type → GPU (L4 / A100; a T4 works but is slower). Put
# `movies_wiki.csv` and `ratings.dat` in your Drive folder and set the paths in the next-but-one cell. Then
# Runtime → Run all. Keep the tab open: checkpoints live on the Colab disk.

# %%
import sys

already_loaded = "transformers" in sys.modules   # an old version in memory would mix with the new files
GENREC_REPO = "https://github.com/mazprimrez/project-playground"

!nvidia-smi --query-gpu=name,memory.total --format=csv
# clean reinstall of the versions the code was tested with (Colab ships its own), then the genrec package
%pip install -q --force-reinstall --no-deps transformers==5.18.0 accelerate==1.15.0 peft==0.21.1
%pip install -q "genrec[train] @ git+{GENREC_REPO}#subdirectory=genrec"

assert not already_loaded, "transformers was already imported: Runtime -> Restart session, then Run all again"

# %%
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
# your project folder on Drive (Files panel -> right-click the folder -> Copy path)
DRIVE_DIR = Path("/content/drive/MyDrive/Self-learning AI related/GenRec - Netflix")
DATA_DIR = DRIVE_DIR                                       # movies_wiki.csv + ratings.dat
START_MODEL = "Qwen/Qwen2.5-0.5B"                          # or DRIVE_DIR / "<previous model>" / "final" to continue
FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-new" / "final"
CHECKPOINT_DIR = Path("/content/checkpoints/genrec")       # Colab disk: a full checkpoint is ~6 GB

for p in [DATA_DIR / "movies_wiki.csv", DATA_DIR / "ratings.dat"]:
    assert p.exists(), f"missing {p}"
assert str(START_MODEL).startswith("Qwen/") or (Path(START_MODEL) / "config.json").exists(), f"no model in {START_MODEL}"

# %% [markdown]
# ## 1. Config

# %%
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from genrec.prompts import build_samples
from genrec.training import (AnswerOnlyTrainer, ChatDataset, PadCollator, TrainConfig, fact_recall, load_model,
                             show_generations)

assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> GPU"
RUN_TRAINING = True   # False: build data + model and run the checks only

cfg = TrainConfig(
    model=str(START_MODEL), data_dir=DATA_DIR, output_dir=CHECKPOINT_DIR,
    next_per_user=20,     # recommendation samples per user (v1: 5, v2: 20)
    epochs=2,             # v1 from the base model: 4; continuing from a trained model: 2
    lr=2e-5,              # 5e-5 from the base model, 2e-5 when continuing
    batch_size=4, grad_accum=8,
)
set_seed(cfg.seed)
cfg

# %% [markdown]
# ## 2. Samples and tokenization

# %%
train_samples, eval_samples = build_samples(cfg)
print("train:", pd.Series([s["task"] for s in train_samples]).value_counts().to_dict())
print("eval: ", {k: len(v) for k, v in eval_samples.items()})

tokenizer = AutoTokenizer.from_pretrained(cfg.model)
train_ds = ChatDataset(train_samples, tokenizer, cfg.max_length)
eval_ds = {k: ChatDataset(v, tokenizer, cfg.max_length) for k, v in eval_samples.items()}
train_ds.describe("train")
for k, v in eval_ds.items():
    v.describe(f"eval/{k}")

# %% [markdown]
# ## 3. Model

# %%
device = "cuda"
# T4 (compute capability 7.5): fp16 mixed precision; L4 / A100 (8.0+): bf16
precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
model = load_model(cfg).to(device)
collator = PadCollator(tokenizer.pad_token_id)

batch = {k: v.to(device) for k, v in collator([train_ds[i] for i in range(cfg.batch_size)]).items()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
    loss = model(**batch).loss
print(f"forward pass {tuple(batch['input_ids'].shape)} in {precision}: loss {loss.item():.3f}")
if not torch.isfinite(loss):   # fp16 can overflow on some models: fall back to full precision
    precision = "fp32"
    print("loss is not finite in fp16 -> training in fp32 (slower)")

picks = [next(s for s in train_samples if s["task"] == t) for t in dict.fromkeys(s["task"] for s in train_samples)]
picks += [v[0] for v in eval_samples.values()]
show_generations(model, tokenizer, picks, device, max_new_tokens=40)

# %% [markdown]
# ## 4. Trainer
#
# The checkpoint with the lowest validation loss on the held-out next-movie users is kept at the end.

# %%
best_metric = "eval_next_movie_loss" if "next_movie" in eval_ds else None
training_args = TrainingArguments(
    output_dir=str(cfg.output_dir), num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr, lr_scheduler_type="cosine",
    warmup_steps=cfg.warmup_ratio,   # transformers v5: a float in [0, 1) is a ratio of total steps
    weight_decay=cfg.weight_decay, logging_steps=cfg.logging_steps,
    eval_strategy="steps" if eval_ds else "no", eval_steps=cfg.eval_steps,
    save_strategy="steps", save_steps=cfg.save_steps, save_total_limit=2,
    load_best_model_at_end=best_metric is not None, metric_for_best_model=best_metric,
    fp16=precision == "fp16", bf16=precision == "bf16", gradient_checkpointing=cfg.gradient_checkpointing,
    train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
    dataloader_num_workers=2, remove_unused_columns=False, report_to="none", seed=cfg.seed,
)
trainer = AnswerOnlyTrainer(model=model, args=training_args, train_dataset=train_ds,
                            eval_dataset=eval_ds or None, data_collator=collator, processing_class=tokenizer)
trainer.create_optimizer()
steps = -(-len(train_ds) // (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
print(f"{steps:.0f} optimizer steps (effective batch {cfg.batch_size * cfg.grad_accum}, lr {cfg.lr}, {precision})")

# the answer-only loss must match the model's own loss (eval mode: no LoRA dropout; same precision for both)
model.eval()
with torch.no_grad(), trainer.accelerator.autocast():
    ours, ref = trainer.compute_loss(model, batch).item(), model(**batch).loss.item()
print(f"loss check: answer-only {ours:.4f} vs default {ref:.4f}")
assert abs(ours - ref) < 1e-2

# memory check: forward + backward on the longest samples (the trainer starts with those) + the AdamW state
longest = sorted(range(len(train_ds)), key=lambda i: len(train_ds.items[i]["input_ids"]))[-cfg.batch_size:]
big = {k: v.to(device) for k, v in collator([train_ds[i] for i in longest]).items()}
model.train()
torch.cuda.reset_peak_memory_stats()
trainer.compute_loss(model, big).backward()
model.zero_grad(set_to_none=True)
adam_gb = 2 * 4 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 2**30
need = torch.cuda.max_memory_allocated() / 2**30 + adam_gb
have = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"memory: ~{need:.1f} GB needed (incl. {adam_gb:.1f} GB optimizer state), GPU has {have:.1f} GB")
del big
torch.cuda.empty_cache()
assert need < 0.85 * have, "won't fit: halve cfg.batch_size and double cfg.grad_accum, or set lora=True"

# %% [markdown]
# ## 5. Train
#
# Re-running this cell after an interruption resumes from the last checkpoint (if the runtime wasn't reset).

# %%
if RUN_TRAINING:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from_checkpoint=True if any(cfg.output_dir.glob("checkpoint-*")) else None)
else:
    print("RUN_TRAINING is False - skipping training")

# %% [markdown]
# ## 6. After training: validation loss, examples, catalogue recall

# %%
if RUN_TRAINING:
    if eval_ds:
        print(trainer.evaluate())
    log = pd.DataFrame(trainer.state.log_history)
    display(log.dropna(subset=["loss"])[["step", "loss"]].tail())
    show_generations(model, tokenizer, picks, device)
    display(fact_recall(model, tokenizer, train_samples, device).to_frame("trained on").style.format("{:.0%}"))

# %% [markdown]
# ## 7. Save the model
#
# Then evaluate it with `eval_qwen_colab.ipynb` (`MODEL_DIR` = this `FINAL_DIR`).

# %%
if RUN_TRAINING:
    final = model.merge_and_unload() if cfg.lora else model   # LoRA: fold the adapters into the base weights
    final = final.to(torch.bfloat16)                          # Qwen's native dtype; ~1 GB
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"saved to {FINAL_DIR}")
