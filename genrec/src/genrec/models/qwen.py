"""The fine-tuned Qwen recommender at inference time: recommendations via constrained beam search, and free-form
answers about movies. Used by the evaluation notebook and the serving endpoint.

Recommendations: the model reads the same prompt it was trained on (`genrec.prompts.next_movie_prompt`) and beam
search proposes `num_beams` titles. Decoding is restricted to a token trie of the catalogue, so every candidate is
a real movie; movies the user already rated are skipped and the first k remaining are returned.
"""
from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import history as split_history
from ..data import target as split_target
from ..metrics import evaluate_lists
from ..prompts import END, catalog_maps, next_movie_prompt, prompt_text


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str):
    """L4 / A100 (compute capability 8+): bf16; T4: fp16; Apple GPU / CPU: fp32."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


class TitleTrie:
    """Token trie of every catalogue title (tokenized exactly like the training answers: title + <|im_end|>)."""

    def __init__(self, tokenizer, titles):
        self.end_id = tokenizer.convert_tokens_to_ids(END)
        self.root = {}
        for title in titles:
            node = self.root
            for tok in tokenizer(title + END, add_special_tokens=False).input_ids:
                node = node.setdefault(tok, {})

    def allowed(self, generated: list) -> list:
        """Token ids that can follow `generated` and still spell a catalogue title."""
        node = self.root
        for tok in generated:
            node = node.get(tok)
            if node is None:
                return [self.end_id]
        return list(node) or [self.end_id]


class QwenRecommender:
    def __init__(self, model_name_or_path, movies, device: str | None = None, dtype=None, history_len: int = 20):
        """model_name_or_path: a local folder or a Hugging Face Hub repo id with the fine-tuned model.
        movies: the catalogue (genrec.data.load_movies) - titles, genres, MovieIDs."""
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype or default_dtype(self.device)).to(self.device).eval()
        self.history_len = history_len
        self.display, self.genres = catalog_maps(movies)
        self.movie_of = {t: m for m, t in self.display.items()}
        self.trie = TitleTrie(self.tokenizer, self.display.values())
        self.end_id = self.trie.end_id

    def prompt(self, history_ids: list) -> str:
        return next_movie_prompt(history_ids[-self.history_len:], self.display, self.genres)

    @torch.no_grad()
    def generate_titles(self, prompts: list, num_beams: int = 30) -> list:
        """num_beams candidate titles per prompt, best first."""
        tok = self.tokenizer
        side, tok.padding_side = tok.padding_side, "left"   # decoder-only batch generation: pad on the left
        try:
            enc = tok([prompt_text(tok, p) for p in prompts], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(self.device)
        finally:
            tok.padding_side = side
        start = enc.input_ids.shape[1]
        out = self.model.generate(
            **enc, max_new_tokens=48, num_beams=num_beams, num_return_sequences=num_beams, do_sample=False,
            use_cache=True, prefix_allowed_tokens_fn=lambda _, ids: self.trie.allowed(ids[start:].tolist()),
            eos_token_id=self.end_id, pad_token_id=tok.pad_token_id)
        titles = [t.strip() for t in tok.batch_decode(out[:, start:], skip_special_tokens=True)]
        return [titles[i * num_beams:(i + 1) * num_beams] for i in range(len(prompts))]

    def recommend(self, histories: list, k: int = 10, num_beams: int = 30, batch_size: int = 8) -> list:
        """histories: MovieIDs oldest first (one list per user) -> top-k MovieIDs per user, already-rated skipped."""
        results = []
        for b in range(0, len(histories), batch_size):
            batch = histories[b:b + batch_size]
            for hist, titles in zip(batch, self.generate_titles([self.prompt(h) for h in batch], num_beams)):
                rated, top = set(hist), []
                for title in titles:
                    m = self.movie_of.get(title)
                    if m is not None and m not in rated and m not in top:
                        top.append(m)
                    if len(top) == k:
                        break
                results.append(top)
        return results

    def title_probabilities(self, history_ids: list, titles: list) -> list:
        """exp(title_logprobs): P(model answers exactly this title | the user's history), in float64."""
        return [math.exp(lp) for lp in self.title_logprobs(history_ids, titles)]

    @torch.no_grad()
    def title_logprobs(self, history_ids: list, titles: list) -> list:
        """log P(model answers exactly this title | the user's history) for each title: the sum of the log-probabilities
        of the title's tokens and the end-of-answer token, tokenized exactly like the training answers. Only the
        answer positions go through the 151k-vocabulary output layer (all positions would need GBs of memory)."""
        tok = self.tokenizer
        prompt_ids = tok(prompt_text(tok, self.prompt(history_ids)), add_special_tokens=False).input_ids
        answers = [tok(t + END, add_special_tokens=False).input_ids for t in titles]
        seqs = [prompt_ids + a for a in answers]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)], mask[i, :len(s)] = torch.tensor(s), 1
        hidden = self.model.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        probs = []
        for i, a in enumerate(answers):
            # the hidden state at position t predicts token t + 1: the answer tokens sit at len(prompt) ...
            pos = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(a), device=self.device)
            logp = torch.log_softmax(self.model.lm_head(hidden[i, pos]).float(), dim=-1)
            rows, cols = torch.arange(len(a), device=self.device), torch.tensor(a, device=self.device)
            probs.append(float(logp[rows, cols].double().sum()))
        return probs

    @torch.no_grad()
    def ask(self, question: str, max_new_tokens: int = 160) -> str:
        """Free-form answer, e.g. 'What is the movie "Heat (1995)" about?' (greedy decoding)."""
        enc = self.tokenizer(prompt_text(self.tokenizer, question), return_tensors="pt",
                             add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                  eos_token_id=self.end_id, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate(recommender: QwenRecommender, sequences, users: list, split: str = "test", k: int = 10,
             num_beams: int = 30, batch_size: int = 8, progress=None) -> tuple:
    """Next-movie metrics for `users` (UserIDs) -> (metrics, {user: top-k MovieIDs}).
    Users are processed shortest prompt first (less padding per batch); results don't depend on the order."""
    order = sorted(users, key=lambda u: len(recommender.prompt(split_history(sequences[u], split))))
    batches = range(0, len(order), batch_size)
    tops = {}
    for b in (progress(batches) if progress else batches):
        chunk = order[b:b + batch_size]
        for u, top in zip(chunk, recommender.recommend([split_history(sequences[u], split) for u in chunk],
                                                       k=k, num_beams=num_beams, batch_size=batch_size)):
            tops[u] = top
    metrics = evaluate_lists([tops[u] for u in users], [split_target(sequences[u], split) for u in users])
    return metrics, tops
