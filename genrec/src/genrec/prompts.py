"""The chat-format training samples and prompts. Training and serving must build prompts exactly the same way -
everything that defines the text the model sees lives here.

Every sample is a chat turn (system / user / assistant); the loss is computed on the assistant answer only.

| task       | question                                         | answer                                   |
|------------|--------------------------------------------------|------------------------------------------|
| plot       | What is the movie "<title>" about?               | Wikipedia plot (first `plot_words` words) |
| genres     | Which genres does "<title>" belong to?           | MovieLens genres                         |
| details    | Who made "<title>" and who stars in it?          | director + main cast                     |
| identify   | Which movie is this? <plot snippet>              | title (year)                             |
| next_movie | A user rated these movies, oldest first ... next? | title (year)                             |
"""
from __future__ import annotations

import random
import re

from .data import history, load_movies, load_ratings, user_sequences

SYSTEM = "You are a movie expert who knows the MovieLens catalogue and recommends movies."
END = "<|im_end|>"
ABBREVIATIONS = {"st", "dr", "mr", "mrs", "ms", "jr", "sr", "vs", "lt", "sgt", "capt", "col", "gen", "no"}
METADATA_TASKS = ["plot", "genres", "details", "identify"]
ALL_TASKS = METADATA_TASKS + ["next_movie"]

TEMPLATES = {
    "plot": ['What is the movie "{t}" about?', 'Summarize the plot of "{t}".', 'Tell me the story of the movie "{t}".'],
    "genres": ['Which genres does the movie "{t}" belong to?', 'What kind of movie is "{t}"?', 'List the genres of "{t}".'],
    "details": ['Who made the movie "{t}" and who stars in it?', 'Who directed "{t}" and who is in the cast?'],
    "identify": ["Which movie is this?\n\n{p}", "Name the movie with this plot:\n\n{p}"],
}


def clean_plot(text):
    text = re.sub(r"\[\d+\]", "", str(text))   # citation markers
    return re.sub(r"\s+", " ", text).strip()


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


def sample(task, prompt, answer):
    return {"task": task, "prompt": prompt, "answer": answer}


def genres_text(genres: str) -> str:
    return genres.replace("|", ", ") if isinstance(genres, str) else ""


def metadata_samples(movies, tasks, cfg, rng):
    """plot / genres / details / identify samples; cfg needs plot_words, snippet_words, max_cast."""
    out = []
    for row in movies.to_dict("records"):
        t = row["display"]
        plot = clean_plot(row["Plot"]) if isinstance(row["Plot"], str) else None
        pick = lambda task: rng.choice(TEMPLATES[task])
        if "genres" in tasks and genres_text(row["Genres"]):
            out.append(sample("genres", pick("genres").format(t=t), genres_text(row["Genres"])))
        if "plot" in tasks and plot:
            out.append(sample("plot", pick("plot").format(t=t), truncate_words(plot, cfg.plot_words)))
        if "identify" in tasks and plot:
            snippet = mask_title(truncate_words(plot, cfg.snippet_words), t)
            out.append(sample("identify", pick("identify").format(p=snippet), t))
        if "details" in tasks:
            parts = []
            if is_known(row["Director"]):
                parts.append(f"Directed by {row['Director'].strip()}.")
            if is_known(row["Cast"]):
                cast = [c.strip() for c in re.split(r",|\n", row["Cast"]) if c.strip()][: cfg.max_cast]
                parts.append(f"Starring {', '.join(cast)}.")
            if parts:
                out.append(sample("details", pick("details").format(t=t), " ".join(parts)))
    return out


def next_movie_prompt(history_ids, display: dict, genres: dict) -> str:
    """The recommendation prompt. history_ids: MovieIDs oldest first; display / genres: MovieID -> text."""
    lines = []
    for k, m in enumerate(history_ids, 1):
        g = genres.get(m, "")
        lines.append(f"{k}. {display[m]} - {g}" if g else f"{k}. {display[m]}")
    return ("Here are the movies a user rated, oldest first:\n" + "\n".join(lines)
            + "\n\nWhich movie will they rate next? Answer with the title and year.")


def catalog_maps(movies):
    """MovieID -> display title, MovieID -> 'Genre, Genre' (the two lookups next_movie_prompt needs)."""
    return (dict(zip(movies.MovieID, movies.display)),
            {m: genres_text(g) for m, g in zip(movies.MovieID, movies.Genres)})


def next_movie_samples(data_dir, movies, cfg, rng):
    """Leave-last-out: train targets come from items[:-2]; eval target is items[-2]; items[-1] is never used.
    cfg needs history_len, min_history, next_per_user, next_eval_users."""
    seqs = user_sequences(load_ratings(data_dir))
    display, genres = catalog_maps(movies)

    train, val = [], []
    for items in seqs:
        train_items = history(items, "val")
        positions = list(range(cfg.min_history, len(train_items)))
        for t in rng.sample(positions, min(cfg.next_per_user, len(positions))):
            hist = train_items[max(0, t - cfg.history_len):t]
            train.append(sample("next_movie", next_movie_prompt(hist, display, genres), display[train_items[t]]))
        if len(train_items) >= 1:
            hist = train_items[-cfg.history_len:]
            val.append(sample("next_movie", next_movie_prompt(hist, display, genres), display[items[-2]]))
    rng.shuffle(val)
    return train, val[: cfg.next_eval_users]


def build_samples(cfg):
    """(train samples, {"metadata": [...], "next_movie": [...]}) for cfg.tasks, deterministic in cfg.seed."""
    rng = random.Random(cfg.seed)
    movies = load_movies(cfg.data_dir)
    meta = metadata_samples(movies, cfg.tasks, cfg, rng)
    rng.shuffle(meta)
    n_eval = int(len(meta) * cfg.eval_frac) if meta else 0
    train, evals = meta[n_eval:], {}
    if n_eval:
        evals["metadata"] = meta[:n_eval]
    if "next_movie" in cfg.tasks:
        nm_train, nm_val = next_movie_samples(cfg.data_dir, movies, cfg, rng)
        train += nm_train
        evals["next_movie"] = nm_val
    rng.shuffle(train)
    return train, evals


def prompt_text(tokenizer, prompt: str) -> str:
    """System + user turn, ending with the assistant header - what the model continues from."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
