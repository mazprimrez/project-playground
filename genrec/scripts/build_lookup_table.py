"""Build the recommendation lookup tables for the API (artifacts/lookup/): the 1,000 test users of ML-1M, each
model's top-10 with probabilities, the users' histories and the movie they actually rated next.

    python scripts/build_lookup_table.py

Inputs (models that are missing are skipped with a note):
    random, most popular   computed here
    sasrec                 artifacts/sasrec-200        (notebooks/movielens/sasrec.ipynb)
    tfrs-sequential        artifacts/lookup_inputs/tfrs_sequential.csv   (notebooks/movielens/tfrs.ipynb)
    qwen                   artifacts/lookup_inputs/qwen.csv   (notebooks/movielens/score_qwen_colab.ipynb, on Colab)

The tables contain MovieLens-derived data (user histories): keep them out of public places - MovieLens may not be
redistributed. They are git-ignored; to serve them from a Space, use a private Hugging Face dataset repo.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from genrec import lookup
from genrec.data import Dataset, history, sample_test_users, target
from genrec.models import baselines as B
from genrec.paths import ARTIFACTS_DIR, ML1M_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ML1M_DIR)
    p.add_argument("--sasrec", type=Path, default=ARTIFACTS_DIR / "sasrec-200")
    p.add_argument("--tfrs", type=Path, default=ARTIFACTS_DIR / "lookup_inputs" / "tfrs_sequential.csv")
    p.add_argument("--qwen", type=Path, default=ARTIFACTS_DIR / "lookup_inputs" / "qwen.csv")
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "lookup")
    args = p.parse_args()

    ds = Dataset.load(args.data_dir)
    seqs = ds.encoded()
    users = sample_test_users(ds.users)
    hist = [history(seqs[u], "test") for u in users]
    decode = lambda i: ds.index.movie_ids[i - 1]
    X = B.interaction_matrix([history(seqs[u], "test") for u in ds.users], ds.index.n_cols)

    parts = []
    scores = B.random_scores(len(users), ds.index.n_cols)              # the same draws as baselines.ipynb
    parts.append(lookup.to_rows("random", users, lookup.normalized_top_k(scores, np.ones_like(scores), hist), decode))
    scores = B.popularity_scores(X, len(users))
    parts.append(lookup.to_rows("most popular", users, lookup.normalized_top_k(scores, scores, hist), decode))

    if (args.tfrs).exists():
        parts.append(pd.read_csv(args.tfrs).assign(model="tfrs-sequential"))
    else:
        print(f"skip tfrs-sequential: no {args.tfrs}")

    if (args.sasrec / "model.pt").exists():
        from genrec.models import sasrec
        model, movie_ids = sasrec.load(args.sasrec)
        assert movie_ids == ds.index.movie_ids, "SASRec was trained on a different catalogue"
        parts.append(lookup.to_rows("sasrec", users, sasrec.recommend_with_probs(model, hist, "cpu"), decode))
    else:
        print(f"skip sasrec: no {args.sasrec / 'model.pt'}")

    if args.qwen.exists():
        parts.append(pd.read_csv(args.qwen).assign(model="qwen"))
    else:
        print(f"skip qwen: no {args.qwen}")

    cols = ["user_id", "model", "rank", "movie_id", "probability"]
    recommendations = pd.concat([part[cols] for part in parts], ignore_index=True)
    assert set(recommendations.user_id) <= set(users), "a model's file has users outside the 1,000 test users"
    movies = ds.movies.rename(columns={"MovieID": "movie_id", "display": "title", "Genres": "genres", "Year": "year"})
    movies = movies[["movie_id", "title", "genres", "year"]]
    user_table = pd.DataFrame({"user_id": users,
                               "history": [ds.sequences[u][:-1] for u in users],
                               "next_movie_id": [target(ds.sequences[u], "test") for u in users]})
    metrics = lookup.write(args.out, movies, user_table, recommendations)
    print(f"wrote {args.out}: {len(users)} users, {len(recommendations)} recommendations")
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
