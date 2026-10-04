# %% [markdown]
# # Qwen recommendations + probabilities for the lookup API (Google Colab)
#
# Produces `qwen.csv` for `scripts/build_lookup_table.py`: for each of the 1,000 test users, the fine-tuned model's
# top-10 movies and **P(the model answers exactly this title | the user's last 20 movies)**.
#
# The top-10 lists come from the evaluation run (`eval_results_<model>.csv` on Drive, written by
# `eval_qwen_colab.ipynb`), so they are exactly the lists behind the reported HR@10; this notebook only scores them
# (one forward pass per user, a few minutes). Without that file it generates the lists first (~25 min on an L4);
# GPU beam search isn't bit-for-bit repeatable, so regenerated lists score slightly differently (HR@10 0.232 vs 0.236).
#
# **Before you run:** Runtime → Change runtime type → GPU; set the paths below; Runtime → Run all. Then download
# `qwen.csv` from Drive into `genrec/artifacts/lookup_inputs/qwen.csv` and run `python scripts/build_lookup_table.py`.

# %%
import sys

already_loaded = "transformers" in sys.modules
GENREC_REPO = "https://github.com/mazprimrez/project-playground"

!nvidia-smi --query-gpu=name,memory.total --format=csv
%pip install -q --force-reinstall --no-deps transformers==5.18.0 accelerate==1.15.0
%pip install -q "genrec[torch] @ git+{GENREC_REPO}#subdirectory=genrec" tqdm

assert not already_loaded, "transformers was already imported: Runtime -> Restart session, then Run all again"

# %%
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/Self-learning AI related/GenRec - Netflix")
DATA_DIR = DRIVE_DIR                                                   # movies_wiki.csv + ratings.dat
MODEL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-v2" / "final"
EVAL_CSV = DRIVE_DIR / f"eval_results_{MODEL_DIR.parent.name}.csv"     # top-10 lists from the evaluation
OUT_CSV = DRIVE_DIR / "qwen.csv"

for p in [DATA_DIR / "movies_wiki.csv", DATA_DIR / "ratings.dat", MODEL_DIR / "config.json"]:
    assert p.exists(), f"missing {p}"

# %%
import ast
import math

import pandas as pd
import torch
from tqdm.auto import tqdm

from genrec.data import Dataset, history, sample_test_users, target
from genrec.metrics import evaluate_lists
from genrec.models import qwen

assert torch.cuda.is_available(), "no GPU: Runtime -> Change runtime type -> GPU"
ds = Dataset.load(DATA_DIR)
test_users = sample_test_users(ds.users)
rec = qwen.QwenRecommender(MODEL_DIR, ds.movies, device="cuda")

# %% [markdown]
# ## Top-10 lists (from the evaluation run when available)

# %%
if EVAL_CSV.exists():
    ev = pd.read_csv(EVAL_CSV)
    ev["top10"] = ev["top10"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    tops = {int(u): [rec.movie_of[t] for t in titles] for u, titles in zip(ev["user"], ev["top10"])}
    assert set(tops) == set(test_users), "the evaluation file has different users"
    print(f"lists from {EVAL_CSV.name}")
else:
    print(f"{EVAL_CSV.name} not found: generating the lists (~25 min)")
    _, tops = qwen.evaluate(rec, ds.sequences, test_users, "test", num_beams=30, batch_size=8,
                            progress=lambda it: tqdm(it, desc="batches"))

metrics = evaluate_lists([tops[u] for u in test_users], [target(ds.sequences[u], "test") for u in test_users])
print({k: round(v, 3) for k, v in metrics.items()}, "(reported: HR@10 0.236 for v2 - exact with the eval file, ~0.005 off when regenerated)")

# %% [markdown]
# ## Probabilities

# %%
rows = []
for u in tqdm(test_users, desc="users"):
    hist = history(ds.sequences[u], "test")
    titles = [rec.display[m] for m in tops[u]]
    for rank, (m, lp) in enumerate(zip(tops[u], rec.title_logprobs(hist, titles)), 1):
        rows.append({"user_id": u, "rank": rank, "movie_id": m, "probability": math.exp(lp), "log_probability": lp})
out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"saved {len(out)} rows -> {OUT_CSV}")
out.groupby("rank")["probability"].describe()[["mean", "50%", "max"]].round(4)
