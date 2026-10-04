# Archive

Superseded code, kept because it produced the runs reported in the main README:

- `make_colab.py`, `make_eval_nb.py`, `eval_cell.py` - generated the self-contained Colab notebooks in
  `notebooks/movielens/archive/` (the exact notebooks used for the Qwen v1/v2 runs). They refer to the old folder
  layout (`training/`, `dataset/`) and are not meant to be run again; the current notebooks use the `genrec` package.
- `train_qwen_legacy.py` - the first command-line training script (before the answer-only loss, memory fixes and
  length-grouped batches). Use `notebooks/movielens/train_qwen_colab.ipynb` / `genrec.training` instead.
