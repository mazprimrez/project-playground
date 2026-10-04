"""Build training/eval_qwen_colab.ipynb: test-set evaluation (HR@10 / NDCG@10) of the phase-2 model on Colab."""
import json
import uuid

SRC = "training/train_qwen.ipynb"
DST = "training/eval_qwen_colab.ipynb"
D = "scripts/archive"   # where eval_cell.py lives
local = {c.get("id"): "".join(c["source"]) for c in json.load(open(SRC))["cells"]}
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


md("""
# Test-set evaluation of the movie recommender (Google Colab)

Evaluates a fine-tuned model (set `MODEL_NAME` in the setup cell: phase 2 or phase 3) on each user's **last** rating, which was never used
for training or validation:

- the model sees the user's previous 20 ratings and proposes movies with beam search that can only spell real
  catalogue titles; movies the user already rated are skipped
- **HR@10**: how often the real next movie is in the top 10; **NDCG@10**: also rewards ranking it higher
- baseline: recommend the most popular movies the user hasn't rated yet. A useful recommender has to beat it.

**Before you run:** Runtime → Change runtime type → GPU, then Runtime → Run all. Needs on Google Drive:
`MyDrive/GenRec/ml-1m/movies_wiki.csv`, `MyDrive/GenRec/ml-1m/ratings.dat` and the model in
`MyDrive/GenRec/<MODEL_NAME>/final`. Takes roughly 25 min for 1,000 users.
""")

code("""
import sys
already_loaded = "transformers" in sys.modules   # an old version in memory would mix with the new files

!nvidia-smi --query-gpu=name,memory.total --format=csv
# clean reinstall: replaces every file of whatever transformers version Colab ships with
%pip install -q --force-reinstall --no-deps transformers==5.18.0 accelerate==1.15.0
%pip install -q transformers==5.18.0 accelerate==1.15.0   # their dependencies, if missing

assert not already_loaded, "transformers was already imported: Runtime -> Restart session, then Run all again"
""")

code("""
import json
import math
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from google.colab import drive
drive.mount("/content/drive")

DRIVE_DIR = Path("/content/drive/MyDrive/GenRec")
DATA_DIR = DRIVE_DIR / "ml-1m"
MODEL_NAME = "qwen2.5-0.5b-movielens-rec"      # phase 2; "qwen2.5-0.5b-movielens-rec-v2" for phase 3
MODEL_DIR = DRIVE_DIR / MODEL_NAME / "final"
HISTORY_LEN = 20   # same as training (Config.history_len)

assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> GPU"
for p in [DATA_DIR / "movies_wiki.csv", DATA_DIR / "ratings.dat", MODEL_DIR / "config.json"]:
    assert p.exists(), f"missing {p}"
""")

md("## 1. Prompt format (identical to training)")
code(local["cc255aec"])
nm = local["45f9e8bd"]
code(nm[nm.index("def next_movie_prompt"):nm.index("def next_movie_samples")].rstrip())
code(local["3576edfb"][:local["3576edfb"].index("class ChatDataset")].rstrip())

md("## 2. Load the model")
code("""
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # L4/A100: bf16, T4: fp16
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
print(f"{MODEL_DIR.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {dtype}, "
      f"{torch.cuda.get_device_name()}")
""")

md("## 3. Evaluate")
cell = open(f"{D}/eval_cell.py").read()
cell = cell.replace("movies_df = load_movies(cfg.data_dir)", "movies_df = load_movies(DATA_DIR)")
cell = cell.replace('ratings = pd.read_csv(cfg.data_dir / "ratings.dat"', 'ratings = pd.read_csv(DATA_DIR / "ratings.dat"')
cell = cell.replace("seqs[u][:-1][-cfg.history_len:]", "seqs[u][:-1][-HISTORY_LEN:]")
cell = cell.replace("GEN_BATCH = 4       # users per generate() call", "GEN_BATCH = 8       # users per generate() call")
assert "cfg." not in cell
code(cell)

md("## 4. Look at some recommendations")
code("""
results.to_csv(DRIVE_DIR / f"eval_results_{MODEL_NAME}.csv", index=False)   # per-user top 10, kept on Drive
for _, r in results.sample(5, random_state=1).iterrows():
    history = [display_of[m] for m in seqs[r["user"]][:-1][-5:]]
    rank = r["top10"].index(r["target"]) + 1 if r["target"] in r["top10"] else None
    print(f"user {r['user']}  last rated: {'; '.join(history)}")
    print(f"   real next: {r['target']}   {f'HIT at rank {rank}' if rank else 'miss'}")
    print(f"   top 10:    {'; '.join(r['top10'])}\\n")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
print(f"wrote {DST}: {len(cells)} cells")
