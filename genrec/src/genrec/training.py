"""Fine-tuning Qwen on the chat samples from genrec.prompts (full fine-tune or LoRA), as used in the notebooks.

Key choices (each one was needed to make training fit and be fast - see notebooks/movielens):
- loss on the assistant answer only (prompt tokens are labelled -100)
- AnswerOnlyTrainer: the 151k-vocabulary output layer runs only on answer positions (memory + ~1/4 of the compute)
- length-grouped batches, padding to a multiple of 8/32, no gradient checkpointing unless memory is tight
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer

from .paths import ML1M_DIR
from .prompts import ALL_TASKS, END, METADATA_TASKS, prompt_text


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    data_dir: Path = ML1M_DIR
    output_dir: Path = Path("checkpoints/qwen2.5-0.5b-movielens")
    tasks: list = field(default_factory=lambda: METADATA_TASKS + ["next_movie"])
    # data
    plot_words: int = 40         # plot answers: a 1-2 sentence summary (150 words is hard to memorise)
    snippet_words: int = 80      # plot snippet length for the identify task
    max_cast: int = 5
    history_len: int = 20        # next_movie: movies shown in the prompt
    min_history: int = 5         # next_movie: minimum history before a target
    next_per_user: int = 20      # next_movie: training targets per user (5 -> HR@10 0.160, 20 -> 0.236 on ML-1M)
    next_eval_users: int = 500   # next_movie: users in the validation set
    eval_frac: float = 0.0       # fraction of metadata samples held out (0: the next_movie users are the validation)
    max_length: int = 600        # 20-movie histories reach ~510 tokens
    # training
    epochs: float = 2            # more passes over the same data overfit (validation loss rises from epoch 3)
    batch_size: int = 4
    grad_accum: int = 8          # effective batch = batch_size * grad_accum = 32
    lr: float = 2e-5             # full fine-tune; 2e-4 with LoRA
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    eval_steps: int = 1000
    save_steps: int = 1000       # must equal eval_steps to keep the best checkpoint
    logging_steps: int = 10
    group_by_length: bool = True
    gradient_checkpointing: bool = False
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42

    def __post_init__(self):
        unknown = set(self.tasks) - set(ALL_TASKS)
        assert not unknown, f"unknown tasks {sorted(unknown)}; choose from {ALL_TASKS}"
        self.data_dir, self.output_dir = Path(self.data_dir), Path(self.output_dir)


class ChatDataset(Dataset):
    """Tokenized chat samples; labels are -100 on the system/user part so the loss covers only the answer."""

    def __init__(self, samples, tokenizer, max_length):
        prompts = tokenizer([prompt_text(tokenizer, s["prompt"]) for s in samples], add_special_tokens=False).input_ids
        answers = tokenizer([s["answer"] + END for s in samples], add_special_tokens=False).input_ids
        self.items, self.dropped = [], 0
        for s, p, a in zip(samples, prompts, answers):
            if len(p) >= max_length - 8:   # prompt alone fills the context: nothing left to learn from
                self.dropped += 1
                continue
            ids = (p + a)[:max_length]
            labels = ([-100] * len(p) + a)[:max_length]
            self.items.append({"input_ids": ids, "labels": labels, "task": s["task"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return {"input_ids": item["input_ids"], "labels": item["labels"]}

    def describe(self, name="data"):
        lengths = sorted(len(x["input_ids"]) for x in self.items)
        tasks = pd.Series([x["task"] for x in self.items]).value_counts().to_dict()
        print(f"{name}: {len(self)} samples (dropped {self.dropped}), tokens median {lengths[len(lengths) // 2]}, "
              f"p95 {lengths[int(len(lengths) * .95)]}, max {lengths[-1]}, tasks {tasks}")


class PadCollator:
    """Right-pads to a multiple of `multiple` (8 suits GPU tensor cores; 32 keeps Apple's GPU memory cache small)."""

    def __init__(self, pad_id, multiple=8):
        self.pad_id = pad_id
        self.multiple = multiple

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        n = -(-n // self.multiple) * self.multiple
        pad = lambda seq, value: seq + [value] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


class AnswerOnlyTrainer(Trainer):
    """Same loss as the default Trainer, but the 151k-vocab output layer only runs on answer tokens.

    The default computes fp32 logits for every token (~700 MB per copy for a long batch) although only the answer
    is in the loss. Only computing it for the answer saves memory and ~1/4 of the compute.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lm = model.get_base_model() if hasattr(model, "get_base_model") else model
        targets = inputs["labels"][:, 1:]
        keep = targets != -100
        with self.accelerator.autocast():   # calling the inner modules skips accelerate's fp16/bf16 forward wrapper
            hidden = lm.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            logits = lm.lm_head(hidden[:, :-1][keep]).float()
        loss = torch.nn.functional.cross_entropy(logits, targets[keep], reduction="sum")
        # num_items_in_batch = answer tokens over all accumulated micro-batches (how the default loss normalises)
        loss = loss / (num_items_in_batch if num_items_in_batch is not None else keep.sum())
        return (loss, {"logits": logits}) if return_outputs else loss


def load_model(cfg: TrainConfig, dtype=torch.float32):
    """Base or fine-tuned causal LM in `dtype` (fp32 master weights: the Trainer autocasts to fp16/bf16),
    optionally wrapped with LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
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
    print(f"model {cfg.model}: {total / 1e6:.0f}M params, {trainable / 1e6:.1f}M trainable, dtype {dtype}")
    return model


# ---------------------------------------------------------------- checks on a trained model

@torch.no_grad()
def answer(model, tokenizer, prompt, device, max_new_tokens=60):
    model.eval()
    enc = tokenizer(prompt_text(tokenizer, prompt), return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                         eos_token_id=tokenizer.convert_tokens_to_ids(END), pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def show_generations(model, tokenizer, samples, device, max_new_tokens=80):
    for s in samples:
        pred = answer(model, tokenizer, s["prompt"], device, max_new_tokens)
        print(f"--- [{s['task']}] {s['prompt'][:120]!r}\n    expected:  {s['answer'][:120]!r}\n    generated: {pred[:120]!r}")


def director(text):
    return text.split("Directed by ")[1].split(". Starring")[0].rstrip(".").strip() if "Directed by " in text else None


def is_hit(task, pred, gold):
    if task == "genres":
        return {g.strip() for g in pred.split(",")} == {g.strip() for g in gold.split(",")}
    if task in ("identify", "next_movie"):
        return pred == gold
    return director(pred) == director(gold)   # details: director only (cast order varies)


def fact_recall(model, tokenizer, samples, device, n=100, seed=0) -> pd.Series:
    """Exact-match accuracy per metadata task on (up to) n samples per task - does the model know the catalogue?"""
    rng, rows = random.Random(seed), []
    for task in ["genres", "details", "identify"]:
        pool = [s for s in samples if s["task"] == task and (task != "details" or "Directed by" in s["answer"])]
        for s in rng.sample(pool, min(n, len(pool))):
            rows.append({"task": task, "hit": is_hit(task, answer(model, tokenizer, s["prompt"], device), s["answer"])})
    return pd.DataFrame(rows).groupby("task").hit.mean()
