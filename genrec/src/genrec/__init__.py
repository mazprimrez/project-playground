"""genrec: generative (LLM) movie recommendation experiments (part of Project Playground).

Modules
- genrec.data      loading MovieLens-format data, the leave-last-out split, item index
- genrec.prompts   chat samples and prompts (shared by training and serving)
- genrec.metrics   hit@1 / HR@10 / NDCG@10 with full ranking
- genrec.models    baselines (random, popularity, ItemKNN, UserKNN), sasrec, qwen, tfrs_models (needs TensorFlow)
- genrec.training  fine-tuning utilities for the Qwen recommender
- genrec.lookup    pre-computed recommendation tables for the API
"""
__version__ = "0.1.0"
