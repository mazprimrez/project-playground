"""Build the Colab notebooks from training/train_qwen.ipynb (shared data/tokenize code is copied verbatim).

phase 1: training/train_qwen_colab.ipynb        - learn the catalogue (metadata tasks) from the base model
phase 2: training/train_qwen_colab_phase2.ipynb - learn to recommend (next_movie + metadata) from the phase-1 model
phase 3: training/train_qwen_colab_phase3.ipynb - same as phase 2, from the phase-2 model with 4x more recommendation data
"""
import json
import sys
import uuid

PHASE = int(sys.argv[1])

SRC = "training/train_qwen.ipynb"
DST = "training/" + {1: "train_qwen_colab.ipynb", 2: "train_qwen_colab_phase2.ipynb",
                                                            3: "train_qwen_colab_phase3.ipynb"}[PHASE]

local = {c.get("id"): "".join(c["source"]) for c in json.load(open(SRC))["cells"]}
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def must_replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


task_table = local["ea3b2889"].split("| task |")[1].split("\n\n")[0]

if PHASE == 1:
    intro = f"""
# Phase 1: teach Qwen2.5-0.5B the MovieLens-1M catalogue (Google Colab)

Colab version of [`train_qwen.ipynb`](train_qwen.ipynb). Every sample is a chat turn (system / user / assistant);
loss is computed on the assistant answer only. Phase 2 ([`train_qwen_colab_phase2.ipynb`](train_qwen_colab_phase2.ipynb))
continues from this model and teaches it to recommend.

| task |{task_table}

**Before you run:**
1. **Runtime → Change runtime type → GPU** (T4 works; L4 / A100 are faster).
2. Upload `dataset/ml-1m/movies_wiki.csv` to Google Drive at **`MyDrive/GenRec/ml-1m/`**.
   No Drive? Set `USE_DRIVE = False` in the Drive cell below and upload the file when prompted.
3. **Runtime → Run all.** The model is saved to `MyDrive/GenRec/qwen2.5-0.5b-movielens/final` (section 10).
"""
elif PHASE == 3:
    intro = f"""
# Phase 3: more recommendation data (Google Colab)

Continues from the phase-2 model with **4x more `next_movie` examples**: 20 target positions per user instead of 5
(~118,000 recommendation samples instead of ~30,000), 4 epochs, metadata tasks still in the mix. The question it
answers: does the gap to SASRec (which learns from every position of every history) come from training data?

Validation is again 500 held-out users (their 2nd-to-last rating) and the checkpoint with the lowest **Next Movie
Loss** is kept at the end. Drawing 20 targets per user changes the random draws, so these are mostly *different*
users from phase 2's: compare the loss with phase 2's 0.688 only roughly. The real comparison is the test evaluation
(`eval_qwen_colab.ipynb`, the same 1,000 users as every other model).

Expect roughly 6-7 hours on an L4-class GPU (~16,000 steps) - **keep this tab open**: checkpoints live on the Colab
disk and a runtime reset loses the progress.

| task |{task_table}

**Before you run:**
1. **Runtime → Change runtime type → GPU** (L4 / A100; a T4 works but is slower).
2. On Google Drive you need `MyDrive/GenRec/ml-1m/movies_wiki.csv`, `MyDrive/GenRec/ml-1m/ratings.dat` and the
   phase-2 model in `MyDrive/GenRec/qwen2.5-0.5b-movielens-rec/final`.
3. **Runtime → Run all.** The model is saved to `MyDrive/GenRec/qwen2.5-0.5b-movielens-rec-v2/final` (section 10).
   Evaluate it with `eval_qwen_colab.ipynb` (set `MODEL_NAME = "qwen2.5-0.5b-movielens-rec-v2"`).
"""
else:
    intro = f"""
# Phase 2: teach the phase-1 model to recommend (Google Colab)

Starts from the phase-1 model (it already knows the catalogue) and trains it on `next_movie`: *a user rated these
movies, oldest first - which one comes next?* The metadata tasks stay in the mix so it doesn't forget the catalogue.

**Validation = 500 held-out users:** for each, the 2nd-to-last rating is never trained on and is what the model has to
predict (the last rating is reserved for a final test). Their loss is the **Next Movie Loss** column, and the
checkpoint with the lowest one is kept at the end.

| task |{task_table}

**Before you run:**
1. **Runtime → Change runtime type → GPU** (T4 works; L4 / A100 are faster).
2. On Google Drive you need:
   - `MyDrive/GenRec/ml-1m/movies_wiki.csv` and **`MyDrive/GenRec/ml-1m/ratings.dat`**
   - the phase-1 model in `MyDrive/GenRec/qwen2.5-0.5b-movielens/final` (from `train_qwen_colab.ipynb`)
3. **Runtime → Run all.** The model is saved to `MyDrive/GenRec/qwen2.5-0.5b-movielens-rec/final` (section 10).
"""
md(intro + """
After a `CUDA out of memory` error, use **Runtime → Restart session** before running again: the crashed run keeps
its GPU memory until then.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
# versions the code was written and tested against (transformers v5 API)
%pip install -q transformers==5.18.0 peft==0.21.1 accelerate==1.15.0
""")

drive_cell = """
from pathlib import Path

USE_DRIVE = True   # False: upload movies_wiki.csv each session instead (lost when the runtime resets)
DRIVE_DIR = Path("/content/drive/MyDrive/GenRec")

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = DRIVE_DIR / "ml-1m"
    FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens" / "final"
else:
    from google.colab import files
    DATA_DIR = Path("/content/ml-1m")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in files.upload().items():   # pick movies_wiki.csv (+ ratings.dat for next_movie)
        (DATA_DIR / name).write_bytes(data)
    FINAL_DIR = Path("/content/qwen2.5-0.5b-movielens/final")

# checkpoints stay on the Colab disk: a full fine-tune checkpoint (weights + optimizer) is ~6 GB
CHECKPOINT_DIR = Path("/content/checkpoints/qwen2.5-0.5b-movielens")
assert (DATA_DIR / "movies_wiki.csv").exists(), f"upload movies_wiki.csv to {DATA_DIR}"
print("data:", sorted(p.name for p in DATA_DIR.iterdir()))
"""
if PHASE >= 2:
    for old, new in [
        ('    FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens" / "final"',
         '    PHASE1_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens" / "final"\n'
         '    FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-rec" / "final"'),
        ('    FINAL_DIR = Path("/content/qwen2.5-0.5b-movielens/final")',
         '    PHASE1_DIR = Path("/content/qwen2.5-0.5b-movielens/final")   # the phase-1 run in this same session\n'
         '    FINAL_DIR = Path("/content/qwen2.5-0.5b-movielens-rec/final")'),
        ('# pick movies_wiki.csv (+ ratings.dat for next_movie)', '# pick movies_wiki.csv and ratings.dat'),
        ('CHECKPOINT_DIR = Path("/content/checkpoints/qwen2.5-0.5b-movielens")',
         'CHECKPOINT_DIR = Path("/content/checkpoints/qwen2.5-0.5b-movielens-rec")'),
        ('assert (DATA_DIR / "movies_wiki.csv").exists(), f"upload movies_wiki.csv to {DATA_DIR}"',
         'assert (DATA_DIR / "movies_wiki.csv").exists(), f"upload movies_wiki.csv to {DATA_DIR}"\n'
         'assert (DATA_DIR / "ratings.dat").exists(), f"upload ratings.dat to {DATA_DIR}"\n'
         'assert (PHASE1_DIR / "config.json").exists(), (\n'
         '    f"no phase-1 model in {PHASE1_DIR}: run train_qwen_colab.ipynb first (incl. the save cell), or set "\n'
         '    "PHASE1_DIR = \'Qwen/Qwen2.5-0.5B\' to start from the base model (then use epochs=4, lr=5e-5)")'),
    ]:
        drive_cell = must_replace(drive_cell, old, new)
    drive_cell = drive_cell.replace('assert (PHASE1_DIR / "config.json").exists()',
                                    'assert str(PHASE1_DIR) == "Qwen/Qwen2.5-0.5B" or (PHASE1_DIR / "config.json").exists()')
if PHASE == 3:
    for old, new in [
        ('    PHASE1_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens" / "final"',
         '    START_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-rec" / "final"   # the phase-2 model'),
        ('    FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-rec" / "final"',
         '    FINAL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-rec-v2" / "final"'),
        ('    PHASE1_DIR = Path("/content/qwen2.5-0.5b-movielens/final")   # the phase-1 run in this same session',
         '    START_DIR = Path("/content/qwen2.5-0.5b-movielens-rec/final")   # the phase-2 run in this same session'),
        ('    FINAL_DIR = Path("/content/qwen2.5-0.5b-movielens-rec/final")',
         '    FINAL_DIR = Path("/content/qwen2.5-0.5b-movielens-rec-v2/final")'),
        ('CHECKPOINT_DIR = Path("/content/checkpoints/qwen2.5-0.5b-movielens-rec")',
         'CHECKPOINT_DIR = Path("/content/checkpoints/qwen2.5-0.5b-movielens-rec-v2")'),
        ('assert str(PHASE1_DIR) == "Qwen/Qwen2.5-0.5B" or (PHASE1_DIR / "config.json").exists(), (\n'
         '    f"no phase-1 model in {PHASE1_DIR}: run train_qwen_colab.ipynb first (incl. the save cell), or set "\n'
         '    "PHASE1_DIR = \'Qwen/Qwen2.5-0.5B\' to start from the base model (then use epochs=4, lr=5e-5)")',
         'assert (START_DIR / "config.json").exists(), (\n'
         '    f"no phase-2 model in {START_DIR}: run train_qwen_colab_phase2.ipynb first (incl. the save cell)")'),
    ]:
        drive_cell = must_replace(drive_cell, old, new)
code(drive_cell)

code("""
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation; before torch

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> T4 GPU"
METADATA_TASKS = ["plot", "genres", "details", "identify"]
ALL_TASKS = METADATA_TASKS + ["next_movie"]
""")

md("## 1. Config")

config_cell = """
RUN_TRAINING = True   # False: build data + model and run the checks only


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    data_dir: Path = DATA_DIR
    output_dir: Path = CHECKPOINT_DIR
    tasks: list = field(default_factory=lambda: list(METADATA_TASKS))   # add "next_movie" for recommendation
    # data
    plot_words: int = 150        # plot answers are cut to ~this many words
    snippet_words: int = 80      # plot snippet length for the identify task
    max_cast: int = 5
    history_len: int = 20        # next_movie: movies shown in the prompt
    min_history: int = 5         # next_movie: minimum history before a target
    next_per_user: int = 5       # next_movie: training targets sampled per user
    next_eval_users: int = 500   # next_movie: users in the eval set
    eval_frac: float = 0.02      # fraction of metadata samples held out for eval
    max_length: int = 512
    # training
    epochs: float = 2
    batch_size: int = 4          # 8 runs out of memory on a T4 (15 GB) with the full fine-tune
    grad_accum: int = 8          # effective batch = batch_size * grad_accum = 32
    lr: float = None             # default: 2e-5 full fine-tune, 2e-4 LoRA
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 10
    group_by_length: bool = True  # batch samples of similar length: much less padding
    gradient_checkpointing: bool = False   # only needed if you hit CUDA out-of-memory
    lora: bool = False           # a 16 GB GPU fits the full fine-tune; True = LoRA (~9M trainable params, faster)
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42

    def __post_init__(self):
        unknown = set(self.tasks) - set(ALL_TASKS)
        assert not unknown, f"unknown tasks {sorted(unknown)}; choose from {ALL_TASKS}"
        if self.lr is None:
            self.lr = 2e-4 if self.lora else 2e-5


cfg = Config()
set_seed(cfg.seed)
cfg
"""
for old, new in [
    ("    plot_words: int = 150        # plot answers are cut to ~this many words",
     "    plot_words: int = 40         # plot answers: a 1-2 sentence summary (150 words is hard to memorise)"),
    ("    batch_size: int = 4          # 8 runs out of memory on a T4 (15 GB) with the full fine-tune",
     "    batch_size: int = 4          # the memory check (section 6) tells you if it fits; 8 fits on ~20 GB GPUs"),
    ("    lr: float = None             # default: 2e-5 full fine-tune, 2e-4 LoRA",
     "    lr: float = 5e-5             # None: 2e-5 full fine-tune, 2e-4 LoRA"),
    ("    epochs: float = 2", "    epochs: float = 4            # recall is ~95% by then; more only overfits"),
]:
    config_cell = must_replace(config_cell, old, new)
if PHASE >= 2:
    for old, new in [
        ('    model: str = "Qwen/Qwen2.5-0.5B"', '    model: str = str(PHASE1_DIR)   # continue from the phase-1 model'),
        ('    tasks: list = field(default_factory=lambda: list(METADATA_TASKS))   # add "next_movie" for recommendation',
         '    tasks: list = field(default_factory=lambda: METADATA_TASKS + ["next_movie"])   # metadata too: no forgetting'),
        ("    eval_frac: float = 0.02      # fraction of metadata samples held out for eval",
         "    eval_frac: float = 0.0       # all metadata is trained on; the held-out next_movie users are the validation"),
        ("    epochs: float = 4            # recall is ~95% by then; more only overfits",
         "    epochs: float = 2            # the catalogue is already learned in phase 1"),
        ("    lr: float = 5e-5             # None: 2e-5 full fine-tune, 2e-4 LoRA",
         "    lr: float = 2e-5             # lower than phase 1: refine the model, don't re-learn it"),
        ("    max_length: int = 512", "    max_length: int = 600        # 20-movie histories reach ~510 tokens"),
        ("    eval_steps: int = 200", "    eval_steps: int = 250"),
        ("    save_steps: int = 200", "    save_steps: int = 250        # must equal eval_steps to keep the best checkpoint"),
        ("    batch_size: int = 4          # the memory check (section 6) tells you if it fits; 8 fits on ~20 GB GPUs",
         "    batch_size: int = 4          # next_movie prompts are long (~400 tokens); the memory check tells you if it fits"),
    ]:
        config_cell = must_replace(config_cell, old, new)
if PHASE == 3:
    for old, new in [
        ('    model: str = str(PHASE1_DIR)   # continue from the phase-1 model',
         '    model: str = str(START_DIR)    # continue from the phase-2 model'),
        ('    next_per_user: int = 5       # next_movie: training targets sampled per user',
         '    next_per_user: int = 20      # next_movie: training targets per user (phase 2: 5)'),
        ("    epochs: float = 2            # the catalogue is already learned in phase 1",
         "    epochs: float = 4            # 4 passes over ~4x more recommendation data than phase 2"),
        ("    eval_steps: int = 250", "    eval_steps: int = 1000       # ~16,000 steps in total: evaluating every 250 would add ~45 min"),
        ("    save_steps: int = 250        # must equal eval_steps to keep the best checkpoint",
         "    save_steps: int = 1000       # must equal eval_steps to keep the best checkpoint"),
    ]:
        config_cell = must_replace(config_cell, old, new)
code(config_cell)

md("## 2. Build the samples")
code(local["cc255aec"])
code(local["45f9e8bd"])
code(local["d1869664"])

md("## 3. Tokenize (loss on the answer only)")
tok = local["3576edfb"]
tok = must_replace(tok, '"""Pads to a multiple of `multiple` so the GPU memory cache sees a few batch shapes instead of hundreds."""',
                   '"""Pads to a multiple of `multiple` (multiples of 8 suit the GPU\'s tensor cores)."""')
tok = must_replace(tok, "def __init__(self, pad_id, multiple=32):", "def __init__(self, pad_id, multiple=8):")
code(tok)
code(local["a86c67fb"])
code(local["70aed9d5"])

md("## 4. Load the model")
gen = local["eee47259"]
show_generations = gen[gen.index("@torch.no_grad()"):gen.index("\n\n\ndevice =")]
code(f'''
device = "cuda"
# T4 (compute capability 7.5) has no fast bf16 -> fp16 mixed precision; L4 / A100 (8.0+) -> bf16
precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def load_model(cfg):
    # fp32 master weights; the Trainer runs the forward/backward in fp16/bf16
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    model.config.use_cache = False
    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"model {{cfg.model}}: {{total / 1e6:.0f}}M params, {{trainable / 1e6:.1f}}M trainable, "
          f"{{torch.cuda.get_device_name()}}, {{precision}} mixed precision")
    return model


{show_generations}

model = load_model(cfg).to(device)
collator = PadCollator(tokenizer.pad_token_id)
''')

md("## 5. Sanity check before training")
code("""
batch = {k: v.to(device) for k, v in collator([train_ds[i] for i in range(cfg.batch_size)]).items()}
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
    loss = model(**batch).loss
print(f"forward pass on one batch {tuple(batch['input_ids'].shape)} in {precision}: loss {loss.item():.3f} "
      f"({time.time() - t0:.2f}s)")
if not torch.isfinite(loss):   # fp16 can overflow on some models: fall back to full precision
    precision = "fp32"
    print("loss is not finite in fp16 -> training in fp32 (slower)")
""")
code(local["11faa7f9"])

md("## 6. Trainer")
trainer_cell = local["4d887894"]
answer_only = trainer_cell[:trainer_cell.index("\n\n\nsteps_per_epoch")]
answer_only = must_replace(
    answer_only,
    "    is in the loss; on Apple GPUs that transient memory piles up in the allocator cache until it hits the limit.\n",
    "    is in the loss. Only computing it for the answer saves memory and ~1/4 of the compute.\n")
loss_check = """# the answer-only loss must match the model's own loss (eval mode: no LoRA dropout; same precision for both)
model.eval()
with torch.no_grad(), trainer.accelerator.autocast():
    check = {k: v.to(device) for k, v in collator([train_ds[i] for i in range(cfg.batch_size)]).items()}
    ours, ref = trainer.compute_loss(model, check).item(), model(**check).loss.item()
print(f"loss check: answer-only {ours:.4f} vs default {ref:.4f}")
assert abs(ours - ref) < 1e-2   # fp16/bf16 rounding differs slightly between the two"""
code(f'''
{answer_only}


steps_per_epoch = -(-len(train_ds) // (cfg.batch_size * cfg.grad_accum))
print(f"{{steps_per_epoch}} optimizer steps/epoch x {{cfg.epochs}} epochs "
      f"(effective batch {{cfg.batch_size * cfg.grad_accum}}, lr {{cfg.lr}}, {{precision}})")

# phase 2: keep the checkpoint that predicts held-out users best (lowest Next Movie Loss)
best_metric = "eval_next_movie_loss" if "next_movie" in eval_ds else None

training_args = TrainingArguments(
    output_dir=str(cfg.output_dir),
    num_train_epochs=cfg.epochs,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum,
    learning_rate=cfg.lr,
    lr_scheduler_type="cosine",
    warmup_steps=cfg.warmup_ratio,   # transformers v5: a float in [0, 1) is a ratio of total steps
    weight_decay=cfg.weight_decay,
    logging_steps=cfg.logging_steps,
    eval_strategy="steps" if eval_ds else "no",
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=2,              # the best and the latest checkpoint
    load_best_model_at_end=best_metric is not None,
    metric_for_best_model=best_metric,
    fp16=precision == "fp16",
    bf16=precision == "bf16",
    gradient_checkpointing=cfg.gradient_checkpointing,
    train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
    seed=cfg.seed,
)
trainer = AnswerOnlyTrainer(model=model, args=training_args, train_dataset=train_ds,
                            eval_dataset=eval_ds or None, data_collator=collator, processing_class=tokenizer)
trainer.create_optimizer()   # catches optimizer / device issues without taking a step
print(f"optimizer: {{type(trainer.optimizer).__name__}}")

{loss_check}
''')

code("""
# memory check: one forward + backward on the longest samples (the trainer starts with those too),
# plus the AdamW state that is only allocated at the first optimizer step
longest = sorted(range(len(train_ds)), key=lambda i: len(train_ds.items[i]["input_ids"]))[-cfg.batch_size:]
batch = {k: v.to(device) for k, v in collator([train_ds[i] for i in longest]).items()}
model.train()
torch.cuda.reset_peak_memory_stats()
trainer.compute_loss(model, batch).backward()
model.zero_grad(set_to_none=True)
adam_gb = 2 * 4 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 2**30
need = torch.cuda.max_memory_allocated() / 2**30 + adam_gb
have = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"memory: ~{need:.1f} GB needed to train (incl. {adam_gb:.1f} GB optimizer state), GPU has {have:.1f} GB")
del batch
torch.cuda.empty_cache()
assert need < 0.85 * have, ("won't fit: halve cfg.batch_size and double cfg.grad_accum, "
                            "or set cfg.lora = True, or cfg.gradient_checkpointing = True")
""")

md("""
## 7. Train

Checkpoints go to the Colab disk every `save_steps`. If the cell is interrupted, re-running it resumes from the
last checkpoint, as long as the runtime hasn't been reset.
""")
code("""
if RUN_TRAINING:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "run_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, f, indent=2)
    resume = any(cfg.output_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
else:
    print("RUN_TRAINING is False - skipping training")
""")

md("## 8. After training")
code(local["1e6641ad"])

md("""
## 9. Recall and recommendation accuracy

Exact-match accuracy on movies the model was trained on (does it know the catalogue?) and, in phase 2, how often its
first guess is the held-out user's next movie (hit@1; a few percent is normal with ~3,700 movies to choose from).
""")
code("""
@torch.no_grad()
def answer(prompt, max_new_tokens=60):
    model.eval()
    enc = tokenizer(prompt_text(tokenizer, prompt), return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=tokenizer.convert_tokens_to_ids(END), pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def director(text):
    return text.split("Directed by ")[1].split(". Starring")[0].rstrip(".").strip() if "Directed by " in text else None


def is_hit(task, pred, gold):
    if task == "genres":
        return {g.strip() for g in pred.split(",")} == {g.strip() for g in gold.split(",")}
    if task in ("identify", "next_movie"):
        return pred == gold
    return director(pred) == director(gold)   # details: director only (cast order varies)


def recall(samples, n=100, seed=0):
    rng, rows = random.Random(seed), []
    for task in ["genres", "details", "identify"]:
        pool = [s for s in samples if s["task"] == task and (task != "details" or "Directed by" in s["answer"])]
        for s in rng.sample(pool, min(n, len(pool))):
            rows.append({"task": task, "hit": is_hit(task, answer(s["prompt"]), s["answer"])})
    return pd.DataFrame(rows).groupby("task").hit.mean()


if RUN_TRAINING:   # a few minutes: ~300-500 answers generated one by one
    if "next_movie" in eval_samples:
        nm = eval_samples["next_movie"][:200]
        hit1 = sum(is_hit("next_movie", answer(s["prompt"], 30), s["answer"]) for s in nm) / len(nm)
        print(f"next_movie hit@1 on {len(nm)} held-out users: {hit1:.1%}")
    results = {"trained on": recall(train_samples)}
    if "metadata" in eval_samples:
        results["held out"] = recall(eval_samples["metadata"])
    display(pd.DataFrame(results).style.format("{:.0%}"))
""")

md("## 10. Save the model")
code("""
if RUN_TRAINING:
    final = model.merge_and_unload() if cfg.lora else model   # LoRA: fold the adapters into the base weights
    final = final.to(torch.bfloat16)                          # Qwen's native dtype; halves the size (~1 GB)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"saved to {FINAL_DIR}")
""")

md("""
### Use the model later

Copy `FINAL_DIR` from Google Drive to your machine (e.g. into `checkpoints/`), then:

```python
tokenizer = AutoTokenizer.from_pretrained(final_dir)
model = AutoModelForCausalLM.from_pretrained(final_dir).to(device)   # a plain model, also when trained with LoRA
show_generations(model, tokenizer, picks, device)
```
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
print(f"wrote {DST}: {len(cells)} cells")
