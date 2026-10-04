"""Fast tests that need no data."""
import math

import numpy as np

from genrec.data import ItemIndex, display_title, history, target
from genrec.metrics import evaluate_lists, evaluate_scores, rank_metrics, top_k
from genrec.prompts import next_movie_prompt, truncate_words


def test_display_title():
    assert display_title("City of Lost Children, The (Cité des enfants perdus, La) (1995)") == "The City of Lost Children (1995)"
    assert display_title("Toy Story (1995)") == "Toy Story (1995)"
    assert display_title("Mad Max 2 (a.k.a. The Road Warrior) (1981)") == "Mad Max 2 (1981)"


def test_split():
    items = [1, 2, 3, 4, 5]
    assert history(items, "val") == [1, 2, 3] and target(items, "val") == 4
    assert history(items, "test") == [1, 2, 3, 4] and target(items, "test") == 5


def test_rank_metrics():
    assert rank_metrics([7, 8, 9], 7) == (1.0, 1.0, 1.0)
    hit1, hr, ndcg = rank_metrics([7, 8, 9], 9)
    assert (hit1, hr) == (0.0, 1.0) and math.isclose(ndcg, 1 / math.log2(4))
    assert rank_metrics([7, 8, 9], 1) == (0.0, 0.0, 0.0)
    assert evaluate_lists([[1, 2], [3, 4]], [1, 9]) == {"hit@1": 0.5, "HR@10": 0.5, "NDCG@10": 0.5}


def test_top_k_skips_padding_and_rated():
    scores = np.array([[9.0, 5, 4, 3, 2]])
    assert top_k(scores, [[1]], k=2).tolist() == [[2, 3]]
    assert evaluate_scores(scores, [[1]], [2], k=2)["hit@1"] == 1.0


def test_item_index_roundtrip():
    idx = ItemIndex([30, 10, 20])
    assert idx.encode([10, 30]) == [2, 1] and idx.decode([2, 1]) == [10, 30] and idx.n_cols == 4


def test_next_movie_prompt():
    p = next_movie_prompt([1, 2], {1: "A (1990)", 2: "B (1991)"}, {1: "Drama", 2: ""})
    assert p == ("Here are the movies a user rated, oldest first:\n1. A (1990) - Drama\n2. B (1991)\n\n"
                 "Which movie will they rate next? Answer with the title and year.")


def test_truncate_words_ends_on_sentence():
    text = "One two three. Four five six seven eight nine ten."
    assert truncate_words(text, 5) == "One two three."
