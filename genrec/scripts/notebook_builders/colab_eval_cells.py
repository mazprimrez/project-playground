# %% [markdown]
# # Test-set evaluation of the Qwen recommender (Google Colab)
#
# Evaluates a fine-tuned model on each user's **last** rating (never used for training or validation), with the
# `genrec` package: the model sees the user's previous 20 ratings and beam search proposes 30 titles that can only
# spell real catalogue movies; already-rated movies are skipped; HR@10 / NDCG@10 / hit@1 on the same 1,000 users as
# every other model in `results/`. Roughly 25 minutes on an L4.
#
# **Before you run:** Runtime → Change runtime type → GPU; set the paths in the next-but-one cell; Runtime → Run all.

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
MODEL_DIR = DRIVE_DIR / "qwen2.5-0.5b-movielens-v2" / "final"          # the model to evaluate
NUM_BEAMS = 30         # keep 30: the reported numbers use it
BATCH_SIZE = 8         # users per generate() call

for p in [DATA_DIR / "movies_wiki.csv", DATA_DIR / "ratings.dat", MODEL_DIR / "config.json"]:
    assert p.exists(), f"missing {p}"

# %%
import time

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
print(f"{MODEL_DIR.parent.name}: {sum(p.numel() for p in rec.model.parameters()) / 1e6:.0f}M params, "
      f"{rec.model.dtype}, {torch.cuda.get_device_name()}; {len(test_users)} test users")

# %% [markdown]
# ## Evaluate

# %%
t0 = time.time()
metrics, tops = qwen.evaluate(rec, ds.sequences, test_users, "test", num_beams=NUM_BEAMS, batch_size=BATCH_SIZE,
                              progress=lambda it: tqdm(it, desc="batches"))

# popularity baseline on the same users (should be HR@10 0.024 on ML-1M)
popular = pd.Series([m for u in ds.users for m in history(ds.sequences[u], "test")]).value_counts().index.tolist()
pop_tops = [[m for m in popular if m not in set(history(ds.sequences[u], "test"))][:10] for u in test_users]
pop = evaluate_lists(pop_tops, [target(ds.sequences[u], "test") for u in test_users])

print(f"{len(test_users)} test users, {time.time() - t0:.0f}s; "
      f"avg {pd.Series([len(t) for t in tops.values()]).mean():.1f} recommendations each")
pd.DataFrame({"model": metrics, "most popular": pop}).style.format("{:.3f}")

# %% [markdown]
# ## Look at some recommendations

# %%
titles = ds.titles()
rows = [{"user": u, "target": titles[target(ds.sequences[u], "test")], "top10": [titles[m] for m in tops[u]]}
        for u in test_users]
pd.DataFrame(rows).to_csv(DRIVE_DIR / f"eval_results_{MODEL_DIR.parent.name}.csv", index=False)
for u in pd.Series(test_users).sample(5, random_state=1):
    tgt = titles[target(ds.sequences[u], "test")]
    top = [titles[m] for m in tops[u]]
    print(f"user {u}  last rated: {'; '.join(titles[m] for m in history(ds.sequences[u], 'test')[-5:])}")
    print(f"   real next: {tgt}   {f'HIT at rank {top.index(tgt) + 1}' if tgt in top else 'miss'}")
    print(f"   top 10:    {'; '.join(top)}\n")
