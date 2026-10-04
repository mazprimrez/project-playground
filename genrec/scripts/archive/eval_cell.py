# Test-set evaluation: each user's LAST rating (never trained on, never used for validation).
# The model proposes TOP_K movies via beam search that can only spell real catalogue titles; movies the user already
# rated are skipped. HR@10 = target in the top 10; NDCG@10 also rewards ranking it higher. Baseline: most popular movies.
import math

from tqdm.auto import tqdm

EVAL_USERS = 1000   # random users to test - the same 1,000 as sasrec.ipynb and baselines.ipynb
TOP_K = 10
NUM_BEAMS = 30      # candidates per user before removing already-rated movies (keep 30: the reported numbers use it)
GEN_BATCH = 8       # users per generate() call

movies_df = load_movies(cfg.data_dir)
display_of = dict(zip(movies_df.MovieID, movies_df.display))
genres_of = dict(zip(movies_df.MovieID, movies_df.Genres.str.replace("|", ", ")))
# "::"-separated; reading it as ":" with the fast C parser and keeping every other column is ~10x faster
ratings = pd.read_csv(cfg.data_dir / "ratings.dat", sep=":", header=None, usecols=[0, 2, 4, 6],
                      names=["UserID", "MovieID", "Rating", "Timestamp"])
seqs = ratings.sort_values(["UserID", "Timestamp"], kind="stable").groupby("UserID").MovieID.apply(list)
users = random.Random(0).sample(list(seqs.index), min(EVAL_USERS, len(seqs)))

# token trie of every catalogue title (tokenized exactly like the training answers) -> beams can only produce real movies
end_id = tokenizer.convert_tokens_to_ids(END)
trie = {}
for title in movies_df.display:
    node = trie
    for tok in tokenizer(title + END, add_special_tokens=False).input_ids:
        node = node.setdefault(tok, {})


def ranked(scores, rated, k=TOP_K):
    """Top-k titles in order, skipping already-rated movies and duplicates."""
    out = []
    for title in scores:
        if title not in rated and title not in out:
            out.append(title)
        if len(out) == k:
            break
    return out


def hit_ndcg(top, target):
    if target not in top:
        return 0.0, 0.0
    return 1.0, 1 / math.log2(top.index(target) + 2)


@torch.no_grad()
def recommend(prompts):
    tokenizer.padding_side = "left"   # decoder-only batch generation: pad on the left
    texts = [prompt_text(tokenizer, p) for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tokenizer.padding_side = "right"
    start = enc.input_ids.shape[1]

    def allowed(batch_id, ids):
        node = trie
        for tok in ids[start:].tolist():
            node = node.get(tok)
            if node is None:
                return [end_id]
        return list(node) or [end_id]

    model.eval()
    out = model.generate(**enc, max_new_tokens=48, num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAMS,
                         do_sample=False, use_cache=True, prefix_allowed_tokens_fn=allowed,
                         eos_token_id=end_id, pad_token_id=tokenizer.pad_token_id)
    titles = tokenizer.batch_decode(out[:, start:], skip_special_tokens=True)
    return [[t.strip() for t in titles[i * NUM_BEAMS:(i + 1) * NUM_BEAMS]] for i in range(len(prompts))]


# popularity baseline: rating counts without anyone's test item
popular = pd.Series([display_of[m] for items in seqs for m in items[:-1]]).value_counts().index.tolist()

prompt_of = {u: next_movie_prompt(seqs[u][:-1][-cfg.history_len:], display_of, genres_of) for u in users}
users.sort(key=lambda u: len(prompt_of[u]))   # similar lengths per batch: less padding

rows, t0 = [], time.time()
for b in tqdm(range(0, len(users), GEN_BATCH), desc="users", unit_scale=GEN_BATCH):
    batch_users = users[b:b + GEN_BATCH]
    for u, beams in zip(batch_users, recommend([prompt_of[u] for u in batch_users])):
        rated = {display_of[m] for m in seqs[u][:-1]}
        target = display_of[seqs[u][-1]]
        top = ranked(beams, rated)
        hit, ndcg = hit_ndcg(top, target)
        pop_hit, pop_ndcg = hit_ndcg(ranked(popular, rated), target)
        rows.append({"user": u, "target": target, "top10": top, "hit@1": float(top[:1] == [target]),
                     "hr@10": hit, "ndcg@10": ndcg, "pop_hr@10": pop_hit, "pop_ndcg@10": pop_ndcg})

results = pd.DataFrame(rows)
summary = pd.DataFrame({
    "model": [results["hit@1"].mean(), results["hr@10"].mean(), results["ndcg@10"].mean()],
    "most popular": [float("nan"), results["pop_hr@10"].mean(), results["pop_ndcg@10"].mean()],
}, index=["hit@1", "HR@10", "NDCG@10"])
print(f"{len(results)} test users, {time.time() - t0:.0f}s; avg {results.top10.map(len).mean():.1f} recommendations each "
      f"(below {TOP_K}: raise NUM_BEAMS)")
display(summary.style.format("{:.3f}"))
