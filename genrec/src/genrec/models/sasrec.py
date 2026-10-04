"""SASRec (Kang & McAuley, 2018): a small causal transformer over a user's movie-ID sequence that predicts the next
movie. Trained with a full softmax over all movies (cross-entropy) instead of the paper's one-negative BCE loss -
cheap with a few thousand movies, and a clearly stronger baseline.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import mean_metrics, rank_metrics


@dataclass
class SASRecConfig:
    max_len: int = 200       # past ratings the model sees
    dim: int = 64
    blocks: int = 2
    heads: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    patience: int = 20       # stop when validation NDCG@10 hasn't improved for this many epochs
    seed: int = 42


class SASRec(nn.Module):
    def __init__(self, n_items: int, cfg: SASRecConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(n_items + 1, cfg.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(cfg.dim, cfg.heads, dim_feedforward=cfg.dim, dropout=cfg.dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(cfg.blocks))
        self.norm = nn.LayerNorm(cfg.dim)
        causal = torch.triu(torch.full((cfg.max_len, cfg.max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal", causal)
        # small init: with PyTorch's default N(0, 1) embeddings the item scores start around +-8 and training stalls
        for emb in (self.item_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq):
        """seq: (B, L) item indices, left-padded with 0 -> (B, L, dim) hidden state per position."""
        keep = (seq > 0).unsqueeze(-1).float()
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions)) * keep
        for layer in self.layers:
            # causal mask only: with left padding a key-padding mask would leave pad rows with nothing to attend to
            x = layer(x, src_mask=self.causal[:seq.shape[1], :seq.shape[1]]) * keep
        return self.norm(x)

    def scores(self, hidden):
        out = hidden @ self.item_emb.weight.T   # tied input/output embeddings
        out[..., 0] = float("-inf")             # never predict padding
        return out


def left_pad(items, length):
    items = items[-length:]
    return [0] * (length - len(items)) + items


@torch.no_grad()
def score_histories(model: SASRec, histories: list, device) -> torch.Tensor:
    """(len(histories), n_items + 1) next-movie scores; already-rated movies are -inf."""
    model.eval()
    seq = torch.tensor([left_pad(h, model.cfg.max_len) for h in histories], device=device)
    scores = model.scores(model(seq)[:, -1])
    # mask every already-rated movie in one indexing op (a per-user loop is ~6,000 GPU calls per evaluation)
    r_idx = torch.repeat_interleave(torch.arange(len(histories)), torch.tensor([len(h) for h in histories]))
    c_idx = torch.tensor([m for h in histories for m in h], dtype=torch.long)
    scores[r_idx.to(device), c_idx.to(device)] = float("-inf")
    return scores


def evaluate(model: SASRec, histories: list, targets: list, device, k: int = 10, batch: int = 512) -> dict:
    rows = []
    for b in range(0, len(histories), batch):
        scores = score_histories(model, histories[b:b + batch], device)
        top = scores.topk(min(k, scores.shape[1]), dim=1).indices.tolist()
        rows += [rank_metrics(t, g) for t, g in zip(top, targets[b:b + batch])]
    return mean_metrics(rows)


def recommend_with_probs(model: SASRec, histories: list, device, k: int = 10, batch: int = 512) -> list:
    """Top-k (item index, probability) per history. Probability = softmax of the model's scores over the movies the
    user hasn't rated yet: P(next movie = X) according to SASRec."""
    out = []
    for b in range(0, len(histories), batch):
        probs = torch.softmax(score_histories(model, histories[b:b + batch], device), dim=1)
        p, idx = probs.topk(min(k, probs.shape[1]), dim=1)
        out += [list(zip(i, q)) for i, q in zip(idx.tolist(), p.tolist())]
    return out


def recommend(model: SASRec, history: list, device, k: int = 10) -> list:
    """Top-k item indices for one history (item indices, oldest first)."""
    scores = score_histories(model, [history], device)[0]
    k = min(k, int(torch.isfinite(scores).sum()))   # never return padding / already-rated movies
    return scores.topk(k).indices.tolist()


def train(sequences: dict, n_items: int, cfg: SASRecConfig, device, log=print) -> tuple:
    """Train on items[:-2] of every user (a random window per user per epoch, every position predicted), pick the
    epoch with the best validation NDCG@10 on items[-2]. sequences: user -> item indices in order.
    Returns (model with the best weights, history of validation metrics)."""
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    users = list(sequences)
    val_users = [u for u in users if len(sequences[u]) >= 3]
    val_hist = [sequences[u][:-2] for u in val_users]
    val_tgt = [sequences[u][-2] for u in val_users]

    model = SASRec(n_items, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    best, best_state, best_epoch, curve, t0 = -1.0, None, 0, [], time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        trainable = [u for u in users if len(sequences[u]) >= 4]   # fresh list each epoch, then shuffled
        rng.shuffle(trainable)
        total = 0.0
        for b in range(0, len(trainable), cfg.batch_size):
            inputs, targets = [], []
            for u in trainable[b:b + cfg.batch_size]:
                seq = sequences[u][:-2]
                end = rng.randint(min(len(seq), cfg.max_len + 1), len(seq))
                window = seq[max(0, end - cfg.max_len - 1):end]
                inputs.append(left_pad(window[:-1], cfg.max_len))
                targets.append(left_pad(window[1:], cfg.max_len))
            inputs, targets = torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
            logits = model.scores(model(inputs))
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        val = evaluate(model, val_hist, val_tgt, device)
        curve.append({"epoch": epoch, "loss": total, **val})
        if val["NDCG@10"] > best:
            best, best_epoch = val["NDCG@10"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            log(f"epoch {epoch:3d}  loss {total:7.1f}  val HR@10 {val['HR@10']:.3f}  NDCG@10 {val['NDCG@10']:.3f}  "
                f"(best {best:.3f} @ {best_epoch})  {time.time() - t0:.0f}s")
        if epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    log(f"stopped at epoch {epoch}, best validation NDCG@10 {best:.3f} at epoch {best_epoch}, {time.time() - t0:.0f}s, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    return model, curve


def save(model: SASRec, movie_ids: list, path) -> None:
    """Weights + config + the MovieID order of the item index, everything needed to serve the model."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(json.dumps({"config": asdict(model.cfg), "movie_ids": [int(m) for m in movie_ids]}))


def load(path, device="cpu") -> tuple:
    """-> (model, movie_ids) where item index i (1..n) is movie_ids[i - 1]."""
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    cfg = SASRecConfig(**meta["config"])
    model = SASRec(len(meta["movie_ids"]), cfg)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device))
    return model.to(device).eval(), meta["movie_ids"]
