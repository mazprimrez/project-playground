import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { useJson } from "../../lib/api";
import { REPO_URL } from "../index";

// ---------------------------------------------------------------- API types (genrec/serving/app.py)

type Movie = { movie_id: number; title: string; genres: string[]; year: number | null };
type ModelInfo = { model: string; description: string; hit_at_1: number; hr_at_10: number; ndcg_at_10: number };
type User = { user_id: number; n_ratings: number; history: Movie[]; next_movie: Movie };
type Recommendation = { rank: number; movie: Movie; probability: number; is_next_movie: boolean };
type Recommendations = { user_id: number; model: string; hit: boolean; recommendations: Recommendation[] };

const API = "/api/genrec";

/** Display names, and the order the models are shown in: the LLM first, the floor last. */
const MODELS: Record<string, { name: string; kind: string }> = {
  qwen: { name: "Qwen2.5-0.5B", kind: "fine-tuned LLM, reads titles" },
  sasrec: { name: "SASRec", kind: "transformer over movie IDs" },
  "tfrs-sequential": { name: "TFRS (GRU)", kind: "TensorFlow Recommenders" },
  "most popular": { name: "Most popular", kind: "baseline" },
  random: { name: "Random", kind: "baseline (the floor)" },
};
const ORDER = Object.keys(MODELS);
const byOrder = (a: string, b: string) => (ORDER.indexOf(a) + 1 || 99) - (ORDER.indexOf(b) + 1 || 99);
const modelName = (m: string) => MODELS[m]?.name ?? m;
const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

// ---------------------------------------------------------------- page

export default function GenrecPage() {
  const [params, setParams] = useSearchParams();
  const userParam = params.get("user");
  const users = useJson<number[]>(`${API}/users?limit=1000`);
  const models = useJson<ModelInfo[]>(`${API}/models`);

  // no user in the URL yet: start with a random one
  useEffect(() => {
    if (!userParam && users.data?.length) setParams({ user: String(pick(users.data)) }, { replace: true });
  }, [userParam, users.data, setParams]);

  const userId = userParam ? Number(userParam) : null;
  const showUser = (id: number) => setParams({ user: String(id) });

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
        <p className="section-sub">
          See what a user watched, what they actually watched next, and each model's top 10 for them.
        </p>
        <UserPicker
          userId={userId}
          users={users.data}
          onPick={showUser}
        />
      </section>

      {userId !== null && <UserView key={userId} userId={userId} />}
    </div>
  );
}

function pick<T>(xs: T[]): T {
  return xs[Math.floor(Math.random() * xs.length)];
}

// ---------------------------------------------------------------- chart

function ModelChart({ models }: { models: ModelInfo[] }) {
  const rows = [...models].sort((a, b) => b.hr_at_10 - a.hr_at_10);
  const max = Math.max(...rows.map((m) => m.hr_at_10));
  return (
    <>
      <div className="bars" role="list">
        {rows.map((m) => (
          <div key={m.model} className="bar-row" role="listitem" tabIndex={0}
               aria-label={`${modelName(m.model)}: ${pct(m.hr_at_10)} hit rate at 10`}>
            <div className="bar-label">
              {modelName(m.model)}
              <small>{MODELS[m.model]?.kind}</small>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(m.hr_at_10 / max) * 100}%` }} />
              <span className="bar-value">{pct(m.hr_at_10)}</span>
            </div>
            <div className="tip" role="tooltip">
              <strong>{modelName(m.model)}</strong> - {m.description}
              <br />
              In the top 10: <strong>{pct(m.hr_at_10)}</strong> · ranked first: <strong>{pct(m.hit_at_1)}</strong> ·
              NDCG@10: <strong>{m.ndcg_at_10.toFixed(3)}</strong>
            </div>
          </div>
        ))}
      </div>
      <p className="chart-note">Hover or focus a bar for hit@1 and NDCG@10.</p>
    </>
  );
}

// ---------------------------------------------------------------- user picker

function UserPicker({ userId, users, onPick }: {
  userId: number | null; users: number[] | null; onPick: (id: number) => void;
}) {
  const [text, setText] = useState(userId?.toString() ?? "");
  useEffect(() => setText(userId?.toString() ?? ""), [userId]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const id = Number(text);
    if (Number.isInteger(id) && id > 0) onPick(id);
  };

  return (
    <form className="picker" onSubmit={submit}>
      <label htmlFor="user-id">User ID</label>
      <input id="user-id" inputMode="numeric" value={text} onChange={(e) => setText(e.target.value)} />
      <button className="btn" type="submit">Show</button>
      <button className="btn primary" type="button" disabled={!users?.length}
              onClick={() => users && onPick(pick(users))}>
        Random user
      </button>
    </form>
  );
}

// ---------------------------------------------------------------- one user

function UserView({ userId }: { userId: number }) {
  const user = useJson<User>(`${API}/users/${userId}?history=10`);
  const compare = useJson<Recommendations[]>(`${API}/users/${userId}/compare`);

  const lists = useMemo(
    () => (compare.data ? [...compare.data].sort((a, b) => byOrder(a.model, b.model)) : []),
    [compare.data],
  );
  // one scale for every list, so probabilities are comparable across models
  const maxProb = useMemo(
    () => Math.max(1e-9, ...lists.flatMap((l) => l.recommendations.map((r) => r.probability))),
    [lists],
  );

  if (user.error) return <p className="error">{user.error}</p>;
  if (!user.data) return <p className="loading">Loading user {userId}…</p>;
  const u = user.data;
  const hits = lists.filter((l) => l.hit).length;

  return (
    <div className="stack">
      <div className="user-grid">
        <section className="card">
          <h2 className="section-title">User {u.user_id}: their last {u.history.length} movies</h2>
          <p className="section-sub">Most recent first, out of {u.n_ratings.toLocaleString()} ratings in total.</p>
          <ul className="history">
            {[...u.history].reverse().map((m) => (
              <li key={m.movie_id}>
                <span>{m.title}</span>
                <span className="genres">{m.genres.join(", ")}</span>
              </li>
            ))}
          </ul>
        </section>
        <section className="card next-card">
          <div className="eyebrow">What they actually watched next</div>
          <h3>{u.next_movie.title}</h3>
          <p>{u.next_movie.genres.join(", ")}</p>
          <p>
            None of the models saw this rating.{" "}
            {compare.data && (hits
              ? `${hits} of ${lists.length} models have it in their top 10 (highlighted below).`
              : "No model has it in its top 10 for this user - a miss is the usual case.")}
          </p>
        </section>
      </div>

      <section>
        <h2 className="section-title">Each model's top 10</h2>
        <p className="section-sub">
          The bar is the model's probability that this is the next movie - small, because it is spread over ~3,700
          movies.
        </p>
        {compare.error && <p className="error">{compare.error}</p>}
        {!compare.data && !compare.error && <p className="loading">Loading recommendations…</p>}
        <div className="compare">
          {lists.map((l) => <ModelColumn key={l.model} list={l} maxProb={maxProb} />)}
        </div>
      </section>
    </div>
  );
}

function ModelColumn({ list, maxProb }: { list: Recommendations; maxProb: number }) {
  return (
    <div className="card model-col">
      <div className="row">
        <h3>{modelName(list.model)}</h3>
        <span className={`badge ${list.hit ? "hit" : "miss"}`}>{list.hit ? "✓ Hit" : "Miss"}</span>
      </div>
      <p className="desc">{MODELS[list.model]?.kind}</p>
      {list.recommendations.length === 0 && <p className="empty">No recommendations for this user.</p>}
      <ol className="recs">
        {list.recommendations.map((r) => (
          <li key={r.movie.movie_id} className={`rec${r.is_next_movie ? " is-next" : ""}`}>
            <span className="rank">{r.rank}</span>
            <div>
              <div className="title">{r.movie.title}</div>
              <div className="prob" title={`P(next movie) = ${pct(r.probability, 2)}`}>
                <div className="prob-track">
                  <div className="prob-fill" style={{ width: `${(r.probability / maxProb) * 100}%` }} />
                </div>
                <span className="prob-value">{pct(r.probability, r.probability < 0.001 ? 2 : 1)}</span>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
