"""Model plumbing: SASRec save/load, the Qwen title trie and constrained recommendation (tiny random model)."""
import pytest
import torch

from genrec.models import sasrec


def test_sasrec_save_load_roundtrip(tmp_path):
    cfg = sasrec.SASRecConfig(max_len=8, dim=16, epochs=1)
    model = sasrec.SASRec(n_items=5, cfg=cfg)
    sasrec.save(model, [11, 12, 13, 14, 15], tmp_path)
    loaded, movie_ids = sasrec.load(tmp_path)
    assert movie_ids == [11, 12, 13, 14, 15]
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    top = sasrec.recommend(loaded, [1, 2], "cpu", k=3)
    assert len(top) == 3 and not {0, 1, 2} & set(top)   # never padding or already-rated


def test_sasrec_trains(tmp_path):
    seqs = {u: [1 + (u + i) % 6 for i in range(12)] for u in range(20)}
    model, curve = sasrec.train(seqs, n_items=6, cfg=sasrec.SASRecConfig(max_len=8, dim=16, epochs=2, patience=5),
                                device="cpu", log=lambda *_: None)
    assert len(curve) == 2 and {"hit@1", "HR@10", "NDCG@10"} <= set(curve[0])


def test_qwen_recommends_only_catalogue_movies(tiny_qwen):
    import pandas as pd
    from genrec.models.qwen import QwenRecommender

    movies = pd.DataFrame({"MovieID": [1, 2, 3, 4, 5, 6],
                           "display": ["Heat (1995)", "Fargo (1996)", "Alien (1979)", "Big (1988)",
                                       "Jaws (1975)", "Toy Story (1995)"],
                           "Genres": ["Action", "Crime", "Horror", "Comedy", "Thriller", "Animation"]})
    rec = QwenRecommender(tiny_qwen, movies, device="cpu")
    tops = rec.recommend([[1, 2], [3]], k=3, num_beams=6)
    assert len(tops) == 2
    for hist, top in zip([[1, 2], [3]], tops):
        assert 0 < len(top) <= 3 and set(top) <= {1, 2, 3, 4, 5, 6} and not set(top) & set(hist)
    assert isinstance(rec.ask('What is the movie "Heat (1995)" about?', max_new_tokens=5), str)


def test_title_probabilities_match_brute_force(tiny_qwen):
    import pandas as pd
    from genrec.models.qwen import QwenRecommender
    from genrec.prompts import END, prompt_text

    movies = pd.DataFrame({"MovieID": [1, 2, 3], "display": ["Heat (1995)", "Fargo (1996)", "Alien (1979)"],
                           "Genres": ["Action", "Crime", "Horror"]})
    rec = QwenRecommender(tiny_qwen, movies, device="cpu")
    got = rec.title_logprobs([1], ["Fargo (1996)", "Alien (1979)"])
    tok = rec.tokenizer
    for title, p in zip(["Fargo (1996)", "Alien (1979)"], got):
        prompt = tok(prompt_text(tok, rec.prompt([1])), add_special_tokens=False).input_ids
        answer = tok(title + END, add_special_tokens=False).input_ids
        logits = rec.model(torch.tensor([prompt + answer])).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        expected = sum(logp[len(prompt) - 1 + j, t] for j, t in enumerate(answer)).item()
        assert p == pytest.approx(expected, rel=1e-4) and p < 0
    probs = rec.title_probabilities([1], ["Fargo (1996)", "Alien (1979)"])
    assert all(0 <= q < 1 for q in probs)


def test_sasrec_probabilities():
    model = sasrec.SASRec(n_items=5, cfg=sasrec.SASRecConfig(max_len=8, dim=16))
    (top,) = sasrec.recommend_with_probs(model, [[1, 2]], "cpu", k=3)
    assert [i for i, _ in top] == sasrec.recommend(model, [1, 2], "cpu", k=3)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1
    (allp,) = sasrec.recommend_with_probs(model, [[1, 2]], "cpu", k=10)
    assert sum(p for _, p in allp) == pytest.approx(1.0)   # a distribution over the 3 unrated movies
