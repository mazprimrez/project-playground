import { useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { useJson } from "../../lib/api";
import { REPO_URL } from "../index";
import Explorer, { API, MODELS } from "./Explorer";

type ModelInfo = { model: string; description: string; hit_at_1: number; hr_at_10: number; ndcg_at_10: number };

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export default function GenrecPage() {
  const [params, setParams] = useSearchParams();
  const userParam = params.get("user");
  const users = useJson<number[]>(`${API}/users?limit=1000`);
  const models = useJson<ModelInfo[]>(`${API}/models`);

  // no user in the URL yet: start with a random one (the URL keeps it, so a view can be shared)
  useEffect(() => {
    if (!userParam && users.data?.length) {
      setParams({ user: String(users.data[Math.floor(Math.random() * users.data.length)]) }, { replace: true });
    }
  }, [userParam, users.data, setParams]);

  return (
    <div className="stack">
      <div>
        <Link to="/" className="back">← All projects</Link>
        <div className="page-head">
          <h1>GenRec: an LLM that recommends movies</h1>
          <p>
            A small language model (Qwen2.5-0.5B) was fine-tuned to know every movie in MovieLens-1M (plots, cast,
            genres) and to answer “a user watched these movies, what do they watch next?”. Here it is next to a
            standard sequential recommender (SASRec), TensorFlow Recommenders and two baselines, on the same 1,000 test
            users. Each user's last movie was held out: can the model guess it?
          </p>
          <div className="links">
            <a href={`${REPO_URL}/tree/main/genrec`} target="_blank" rel="noreferrer">Code and full results</a>
            <a href={`${API}/docs`}>API docs</a>
          </div>
        </div>
      </div>

      <section className="card">
        <h2 className="section-title">How often is the next movie in the top 10?</h2>
        <p className="section-sub">Hit rate @10 over 1,000 test users, ranking every movie in the catalogue.</p>
        {models.error && <p className="error">{models.error}</p>}
        {models.data && <ModelChart models={models.data} />}
      </section>

      <section>
        <h2 className="section-title">Try it on a user</h2>
        <p className="section-sub">What they watched, what they actually watched next, and every model's top 10.</p>
        <Explorer userId={userParam ? Number(userParam) : null} users={users.data}
                  onPick={(id) => setParams({ user: String(id) })} />
      </section>
    </div>
  );
}

function ModelChart({ models }: { models: ModelInfo[] }) {
  const rows = [...models].sort((a, b) => b.hr_at_10 - a.hr_at_10);
  const max = Math.max(...rows.map((m) => m.hr_at_10));
  return (
    <>
      <div className="bars" role="list">
        {rows.map((m) => {
          const info = MODELS[m.model];
          return (
            <div key={m.model} className="bar-row" role="listitem" tabIndex={0}
                 aria-label={`${info?.name ?? m.model}: ${pct(m.hr_at_10)} hit rate at 10`}>
              <div className="bar-label">
                {info?.name ?? m.model}
                <small>{info?.tag}</small>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(m.hr_at_10 / max) * 100}%` }} />
                <span className="bar-value">{pct(m.hr_at_10)}</span>
              </div>
              <div className="tip" role="tooltip">
                <strong>{info?.name ?? m.model}</strong> - {m.description}
                <br />
                In the top 10: <strong>{pct(m.hr_at_10)}</strong> · ranked first: <strong>{pct(m.hit_at_1)}</strong> ·
                NDCG@10: <strong>{m.ndcg_at_10.toFixed(3)}</strong>
              </div>
            </div>
          );
        })}
      </div>
      <p className="chart-note">Hover or focus a bar for hit@1 and NDCG@10.</p>
    </>
  );
}
