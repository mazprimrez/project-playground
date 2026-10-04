import pytest
import torch

from genrec.paths import ML1M_DIR


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `data` when the datasets haven't been downloaded."""
    if (ML1M_DIR / "ratings.dat").exists() and (ML1M_DIR / "movies_wiki.csv").exists():
        return
    skip = pytest.mark.skip(reason="needs data/ml-1m (run scripts/download_data.py)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def ml1m():
    from genrec.data import Dataset
    return Dataset.load(ML1M_DIR)


@pytest.fixture(scope="session")
def tiny_qwen(tmp_path_factory):
    """Qwen's real tokenizer with a tiny random model, saved like a fine-tuned checkpoint."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:   # offline and not cached
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    path = tmp_path_factory.mktemp("tiny_qwen")
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path
