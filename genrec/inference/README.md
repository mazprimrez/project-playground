# GenRec live inference

The fine-tuned Qwen2.5-0.5B as a web service: send any list of movies from the 4,000-movie catalogue (2010-2023),
get the model's top 10 next movies. Runs on Cloud Run (CPU, service `genrec-model`), so a call takes up to a minute
and the first call after idle also loads the model.

- `POST /recommend` with `{"movie_ids": [79132, 109487, 134130], "k": 10}` (MovieLens IDs, oldest first)
- `GET /health`, interactive docs at `/docs`

The model is downloaded at startup from the private Hugging Face repo
[sparklingdust/genrec-qwen2.5-0.5b-movies](https://huggingface.co/sparklingdust/genrec-qwen2.5-0.5b-movies) with
the `HF_TOKEN` secret. Deployed by `.github/workflows/deploy-model.yml` when this folder changes.
