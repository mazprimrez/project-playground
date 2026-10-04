"""Fine-tune Qwen2.5-0.5B on MovieLens-1M movie metadata (+ optional next-movie prediction).

Every sample is a chat turn (system / user / assistant); loss is computed on the assistant answer only.

Tasks (choose with --tasks):
  plot        'What is the movie "<title>" about?'          -> Wikipedia plot (first --plot-words words)
  genres      'Which genres does "<title>" belong to?'      -> MovieLens genres
  details     'Who made "<title>" and who stars in it?'     -> director + main cast
  identify    'Which movie is this? <plot snippet>'         -> title (year)
  next_movie  'A user rated these movies ... next?'         -> title (year)
              uses ratings.dat with the same leave-last-out split as interaction-analysis.ipynb:
              the last 2 ratings of every user are never trained on (2nd-to-last = eval, last = reserved test)

Examples:
  python training/train_qwen.py --dry-run                     # build data, check masking, one forward pass, no training
  python training/train_qwen.py                               # full fine-tune on the metadata tasks
  python training/train_qwen.py --tasks plot,genres,details,identify,next_movie
  python training/train_qwen.py --lora                        # LoRA instead of full fine-tuning (less memory)
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parent.parent
METADATA_TASKS = ["plot", "genres", "details", "identify"]
ALL_TASKS = METADATA_TASKS + ["next_movie"]
SYSTEM = "You are a movie expert who knows the MovieLens catalogue and recommends movies."
ARTICLES = r"The|A|An|La|Le|Les|L'|Il|El|Das|Der|Die"
END = "<|im_end|>"

TEMPLATES = {
    "plot": ['What is the movie "{t}" about?', 'Summarize the plot of "{t}".', 'Tell me the story of the movie "{t}".'],
    "genres": ['Which genres does the movie "{t}" belong to?', 'What kind of movie is "{t}"?', 'List the genres of "{t}".'],
    "details": ['Who made the movie "{t}" and who stars in it?', 'Who directed "{t}" and who is in the cast?'],
    "identify": ["Which movie is this?\n\n{p}", "Name the movie with this plot:\n\n{p}"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "ml-1m")
    p.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "qwen2.5-0.5b-movielens")
    p.add_argument("--tasks", default=",".join(METADATA_TASKS), help=f"comma-separated subset of {ALL_TASKS}")
    # data
    p.add_argument("--plot-words", type=int, default=150, help="plot answers are cut to ~this many words")
    p.add_argument("--snippet-words", type=int, default=80, help="plot snippet length for the identify task")
    p.add_argument("--max-cast", type=int, default=5)
    p.add_argument("--history-len", type=int, default=20, help="next_movie: movies shown in the prompt")
    p.add_argument("--min-history", type=int, default=5, help="next_movie: minimum history before a target")
    p.add_argument("--next-per-user", type=int, default=5, help="next_movie: training targets sampled per user")
    p.add_argument("--next-eval-users", type=int, default=500, help="next_movie: users in the eval set")
    p.add_argument("--eval-frac", type=float, default=0.02, help="fraction of metadata samples held out for eval")
    p.add_argument("--max-length", type=int, default=512)
    # training
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="default: 2e-5 full fine-tune, 2e-4 LoRA")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="build data + model, show samples, run one forward pass and a few generations; no training")
    args = p.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(args.tasks) - set(ALL_TASKS)
    if unknown:
        p.error(f"unknown tasks {sorted(unknown)}; choose from {ALL_TASKS}")
    if args.lr is None:
        args.lr = 2e-4 if args.lora else 2e-5
    return args


# --------------------------------------------------------------------------- data

def display_title(ml_title):
    """'City of Lost Children, The (Cité des enfants perdus, La) (1995)' -> 'The City of Lost Children (1995)'"""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", ml_title)
    name, year = (m.group(1), m.group(2)) if m else (ml_title, None)
    name = re.sub(r"\s*\([^()]*\)", "", name).strip() or name   # drop alternate titles
    name = re.sub(rf"^(.*), ({ARTICLES})$",
                  lambda a: a[2] + ("" if a[2].endswith("'") else " ") + a[1], name)
    return f"{name} ({year})" if year else name


def clean_plot(text):
    text = re.sub(r"\[\d+\]", "", str(text))   # citation markers
    return re.sub(r"\s+", " ", text).strip()


ABBREVIATIONS = {"st", "dr", "mr", "mrs", "ms", "jr", "sr", "vs", "lt", "sgt", "capt", "col", "gen", "no"}


def truncate_words(text, n):
    """Cut to ~n words, ending on the last full sentence when that keeps at least half the text."""
    words = text.split()
    if len(words) <= n:
        return text
    cut = " ".join(words[:n])
    ends = [m.end() for m in re.finditer(r"[.!?][\"')]?(?=\s)", cut)
            if cut[:m.start()].split()[-1].lower().strip("\"'(") not in ABBREVIATIONS]
    return cut[:ends[-1]] if ends and ends[-1] > len(cut) * 0.5 else cut + " ..."


def mask_title(text, display):
    """Hide the movie's own name in an identify snippet so the answer isn't given away."""
    name = re.sub(r"\s*\(\d{4}\)$", "", display)
    for n in sorted({name, re.sub(r"^(The|A|An) ", "", name)}, key=len, reverse=True):
        if len(n) >= 4:   # skip very short names like "It" / "Big" that are ordinary words
            text = re.sub(rf"\b{re.escape(n)}\b", "___", text)
    return text


def is_known(value):
    return isinstance(value, str) and value.strip() and value.strip().lower() != "unknown"


def load_movies(data_dir):
    movies = pd.read_csv(data_dir / "movies_wiki.csv")
    movies["display"] = movies["Title"].map(display_title)
    # two different movies must never share a display title (it is the answer for identify / next_movie)
    dup = movies["display"].duplicated(keep=False)
    movies.loc[dup, "display"] = movies.loc[dup, "Title"]
    assert movies["display"].is_unique, movies.loc[movies["display"].duplicated(keep=False), "display"]
    return movies


def sample(task, prompt, answer):
    return {"task": task, "prompt": prompt, "answer": answer}


def metadata_samples(movies, tasks, args, rng):
    out = []
    for row in movies.to_dict("records"):
        t = row["display"]
        plot = clean_plot(row["Plot"]) if isinstance(row["Plot"], str) else None
        pick = lambda task: rng.choice(TEMPLATES[task])
        if "genres" in tasks:
            out.append(sample("genres", pick("genres").format(t=t), row["Genres"].replace("|", ", ")))
        if "plot" in tasks and plot:
            out.append(sample("plot", pick("plot").format(t=t), truncate_words(plot, args.plot_words)))
        if "identify" in tasks and plot:
            snippet = mask_title(truncate_words(plot, args.snippet_words), t)
            out.append(sample("identify", pick("identify").format(p=snippet), t))
        if "details" in tasks:
            parts = []
            if is_known(row["Director"]):
                parts.append(f"Directed by {row['Director'].strip()}.")
            if is_known(row["Cast"]):
                cast = [c.strip() for c in re.split(r",|\n", row["Cast"]) if c.strip()][: args.max_cast]
                parts.append(f"Starring {', '.join(cast)}.")
            if parts:
                out.append(sample("details", pick("details").format(t=t), " ".join(parts)))
    return out


def next_movie_prompt(history, display, genres):
    lines = [f"{k}. {display[m]} - {genres[m]}" for k, m in enumerate(history, 1)]
    return ("Here are the movies a user rated, oldest first:\n" + "\n".join(lines)
            + "\n\nWhich movie will they rate next? Answer with the title and year.")


def next_movie_samples(data_dir, movies, args, rng):
    """Leave-last-out split: train targets come from items[:-2]; eval target is items[-2]; items[-1] is never used."""
    ratings = pd.read_csv(data_dir / "ratings.dat", sep="::", engine="python",
                          names=["UserID", "MovieID", "Rating", "Timestamp"])
    seqs = ratings.sort_values(["UserID", "Timestamp"], kind="stable").groupby("UserID").MovieID.apply(list)
    display = dict(zip(movies.MovieID, movies.display))
    genres = dict(zip(movies.MovieID, movies.Genres.str.replace("|", ", ")))

    train, val = [], []
    for items in seqs:
        train_items = items[:-2]
        positions = list(range(args.min_history, len(train_items)))
        for t in rng.sample(positions, min(args.next_per_user, len(positions))):
            history = train_items[max(0, t - args.history_len):t]
            train.append(sample("next_movie", next_movie_prompt(history, display, genres), display[train_items[t]]))
        history = train_items[-args.history_len:]
        val.append(sample("next_movie", next_movie_prompt(history, display, genres), display[items[-2]]))
    rng.shuffle(val)
    return train, val[: args.next_eval_users]


def build_samples(args):
    rng = random.Random(args.seed)
    movies = load_movies(args.data_dir)
    meta = metadata_samples(movies, args.tasks, args, rng)
    rng.shuffle(meta)
    n_eval = int(len(meta) * args.eval_frac) if meta else 0
    train, evals = meta[n_eval:], {}
    if n_eval:
        evals["metadata"] = meta[:n_eval]
    if "next_movie" in args.tasks:
        nm_train, nm_val = next_movie_samples(args.data_dir, movies, args, rng)
        train += nm_train
        evals["next_movie"] = nm_val
    rng.shuffle(train)
    return train, evals


# --------------------------------------------------------------------------- tokenization

def prompt_text(tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ChatDataset(Dataset):
    """Tokenized chat samples; labels are -100 on the system/user part so loss covers only the answer."""

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


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, value: seq + [value] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


# --------------------------------------------------------------------------- model

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args, device):
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if use_bf16 else torch.float32)
    model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        if not args.no_gradient_checkpointing:
            model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"model {args.model}: {total / 1e6:.0f}M params, {trainable / 1e6:.1f}M trainable, "
          f"dtype {next(model.parameters()).dtype}, device {device}")
    return model, use_bf16


@torch.no_grad()
def show_generations(model, tokenizer, samples, device, max_new_tokens=80):
    model.eval()
    for s in samples:
        enc = tokenizer(prompt_text(tokenizer, s["prompt"]), return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.convert_tokens_to_ids(END), pad_token_id=tokenizer.pad_token_id)
        pred = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"--- [{s['task']}] {s['prompt'][:120]!r}\n    expected:  {s['answer'][:120]!r}\n    generated: {pred[:120]!r}")


# --------------------------------------------------------------------------- main

def describe(name, ds):
    lengths = sorted(len(x["input_ids"]) for x in ds.items)
    tasks = pd.Series([x["task"] for x in ds.items]).value_counts().to_dict()
    print(f"{name}: {len(ds)} samples (dropped {ds.dropped}), tokens median {lengths[len(lengths) // 2]}, "
          f"p95 {lengths[int(len(lengths) * .95)]}, max {lengths[-1]}, tasks {tasks}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()

    t0 = time.time()
    train_samples, eval_samples = build_samples(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = ChatDataset(train_samples, tokenizer, args.max_length)
    eval_ds = {k: ChatDataset(v, tokenizer, args.max_length) for k, v in eval_samples.items()}
    describe("train", train_ds)
    for k, v in eval_ds.items():
        describe(f"eval/{k}", v)
    print(f"data ready in {time.time() - t0:.1f}s")

    model, use_bf16 = load_model(args, device)
    collator = PadCollator(tokenizer.pad_token_id)

    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    print(f"{steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs "
          f"(effective batch {args.batch_size * args.grad_accum}, lr {args.lr})")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,   # transformers v5: a float in [0, 1) is a ratio of total steps
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=eval_ds or None, data_collator=collator, processing_class=tokenizer)

    if args.dry_run:
        trainer.create_optimizer()   # catches optimizer / device issues without taking a step
        print(f"optimizer: {type(trainer.optimizer).__name__}")
        dry_run(model, tokenizer, train_ds, train_samples, eval_samples, collator, args, device)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    trainer.train()
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if eval_ds:
        print(trainer.evaluate())
    print(f"saved to {final_dir}")
    show_generations(model, tokenizer, [v[0] for v in eval_samples.values()], device)


def dry_run(model, tokenizer, train_ds, train_samples, eval_samples, collator, args, device):
    print("\n=== DRY RUN: no optimizer steps ===")
    first = {}
    for i, item in enumerate(train_ds.items):
        first.setdefault(item["task"], i)
    for task, i in first.items():
        item = train_ds.items[i]
        answer_ids = [t for t in item["labels"] if t != -100]
        print(f"\n--- {task} ({len(item['input_ids'])} tokens, {len(answer_ids)} in loss)")
        print(tokenizer.decode(item["input_ids"]))
        print(f"[LOSS ON] {tokenizer.decode(answer_ids)!r}")

    model.to(device)
    batch = collator([train_ds[i] for i in range(args.batch_size)])
    batch = {k: v.to(device) for k, v in batch.items()}
    t0 = time.time()
    with torch.no_grad():
        loss = model(**batch).loss
    print(f"\nforward pass on one batch {tuple(batch['input_ids'].shape)}: loss {loss.item():.3f} "
          f"({time.time() - t0:.2f}s)")

    print("\nbase model generations (before training):")
    picks = [next(s for s in train_samples if s["task"] == t) for t in first]
    picks += [v[0] for v in eval_samples.values()]
    show_generations(model, tokenizer, picks, device, max_new_tokens=40)


if __name__ == "__main__":
    main()
