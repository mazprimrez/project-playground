# GenRec live inference

The fine-tuned Qwen2.5-0.5B as a web service: send any list of movies from the 4,000-movie catalogue (2010-2023),
get the model's top 10 next movies. Runs on Cloud Run (CPU, service `genrec-model`), so a call takes up to a minute
and the first call after idle also loads the model.

- `POST /recommend` with `{"movie_ids": [79132, 109487, 134130], "k": 10}` (MovieLens IDs, oldest first)
- add `"explain": true` to also get, for each pick, the movies it depends on most: each of the last 10 movies is
  removed in turn and the pick re-scored ("because of Interstellar -52%"). Several times slower (minutes on Cloud Run;
  the service allows 10-minute requests)
- `GET /health`, interactive docs at `/docs`

Cloud Run settings (kept across deploys): 2 vCPU, 6 GiB, at most 1 instance (caps the cost), scales to zero,
`NUM_THREADS=2` (the container reports more CPUs than it gets; extra threads made it ~1.5x slower) and `NUM_BEAMS=15`
(~30 s per call; 30 beams, as in the evaluation, takes ~40 s and gave the same top 5 in tests).

The model is downloaded at startup from the private Hugging Face repo
[sparklingdust/genrec-qwen2.5-0.5b-movies](https://huggingface.co/sparklingdust/genrec-qwen2.5-0.5b-movies) with
the `HF_TOKEN` secret. Deployed by `.github/workflows/deploy-model.yml` when this folder changes.
