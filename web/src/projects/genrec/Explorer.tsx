import { useEffect, useMemo, useState, type FormEvent } from "react";

import { useJson } from "../../lib/api";
import { genreColor, inkFor, Poster, type Movie } from "./posters";

type User = { user_id: number; n_ratings: number; history: Movie[]; next_movie: Movie };
type Recommendation = { rank: number; movie: Movie; probability: number; is_next_movie: boolean };
type Recommendations = { user_id: number; model: string; hit: boolean; recommendations: Recommendation[] };

export const API = "/api/genrec";
const HISTORY = 20;        // how many recent movies to show (and build the taste profile from)

/** The rows, from the floor up to the LLM. Colors identify the model's probability bars. */
export const MODELS: Record<string, { name: string; tag: string; desc: string; color: string }> = {
  random: { name: "Random", tag: "Baseline", desc: "Unseen movies at random - the floor to beat", color: "var(--m-random)" },
  "most popular": { name: "Popular", tag: "Baseline", desc: "Most-rated movies overall, minus what this user has seen", color: "var(--m-popular)" },
  "tfrs-sequential": { name: "TFRS", tag: "Sequential", desc: "TensorFlow Recommenders: a GRU over the last 20 movies", color: "var(--m-tfrs)" },
  sasrec: { name: "SASRec", tag: "Sequential", desc: "Self-attention over the user's time-ordered ratings", color: "var(--m-sasrec)" },
  qwen: { name: "GenRec", tag: "Generative", desc: "Fine-tuned Qwen2.5-0.5B that writes the next title", color: "var(--m-qwen)" },
};
const ORDER = Object.keys(MODELS);
const rankOf = (m: string) => (ORDER.indexOf(m) + 1 || 99);

const fmtP = (p: number) => (p < 0.001 ? p.toPrecision(2) : p.toFixed(3));

function pick<T>(xs: T[]): T {
  return xs[Math.floor(Math.random() * xs.length)];
}

/** The user's top genres in their recent history, most frequent first. */
function tasteProfile(history: Movie[], n = 4): [string, number][] {
  const counts = new Map<string, number>();
  history.forEach((m) => m.genres.forEach((g) => counts.set(g, (counts.get(g) ?? 0) + 1)));
  return [...counts].sort((a, b) => b[1] - a[1]).slice(0, n);
}

// ---------------------------------------------------------------- the explorer

export default function Explorer({ userId, users, onPick }: {
  userId: number | null; users: number[] | null; onPick: (id: number) => void;
}) {
  const shuffle = () => users?.length && onPick(pick(users.filter((u) => u !== userId)));
  if (userId === null) return <p className="loading">Picking a user…</p>;
  return <UserExplorer key={userId} userId={userId} onShuffle={shuffle} onPick={onPick} canShuffle={!!users?.length} />;
}

function UserExplorer({ userId, onShuffle, onPick, canShuffle }: {
  userId: number; onShuffle: () => void; onPick: (id: number) => void; canShuffle: boolean;
}) {
  const user = useJson<User>(`${API}/users/${userId}?history=${HISTORY}`);
  const compare = useJson<Recommendations[]>(`${API}/users/${userId}/compare`);

  const rows = useMemo(
    () => (compare.data ? [...compare.data].sort((a, b) => rankOf(a.model) - rankOf(b.model)) : []),
    [compare.data],
  );
  // how many models recommend each movie (the ×N badge)
  const picks = useMemo(() => {
    const n = new Map<number, number>();
    rows.forEach((r) => r.recommendations.forEach((x) => n.set(x.movie.movie_id, (n.get(x.movie.movie_id) ?? 0) + 1)));
    return n;
  }, [rows]);
  // one probability scale for every row, so the bars compare across models
  const maxP = useMemo(
    () => Math.max(1e-9, ...rows.flatMap((r) => r.recommendations.map((x) => x.probability))), [rows]);

  if (user.error) {
    return (
      <div className="card">
        <p className="error">{user.error}</p>
        <button className="btn primary" style={{ marginTop: 12 }} onClick={onShuffle} disabled={!canShuffle}>
          Show a random user
        </button>
      </div>
    );
  }
  if (!user.data) return <p className="loading">Loading user {userId}…</p>;

  const u = user.data;
  const taste = tasteProfile(u.history);
  const tasteSet = new Set(taste.map(([g]) => g));
  const recent = [...u.history].reverse();
  const found = rows.filter((r) => r.hit)
    .map((r) => ({ model: r.model, rank: r.recommendations.find((x) => x.is_next_movie)!.rank }));

  return (
    <div className="explorer">
      <UserHeader user={u} taste={taste} onShuffle={onShuffle} onPick={onPick} canShuffle={canShuffle} />

      <section>
        <div className="row-head">
          <h3>Rated by #{u.user_id}</h3>
          <span className="row-desc">their last {u.history.length} movies, most recent first</span>
        </div>
        <div className="poster-row">
          {recent.map((m, i) => (
            <Poster key={m.movie_id} movie={m}><span className="foot-rank">{i === 0 ? "latest" : `${i + 1} back`}</span></Poster>
          ))}
        </div>
      </section>

      <section className="card spotlight">
        <Poster movie={u.next_movie} highlight badges={<span className="pill pill-next">✓ Watched next</span>} />
        <div className="spotlight-text">
          <div className="eyebrow">What they actually watched next</div>
          <h3>{u.next_movie.title}</h3>
          <p>{u.next_movie.genres.join(" · ")}</p>
          <p className="spotlight-note">
            Held out: none of the models saw this rating. Each one had to guess it from the history above.
          </p>
          {compare.data && (
            found.length ? (
              <ul className="found-list">
                {found.map((f) => (
                  <li key={f.model}>
                    <span className="dot" style={{ background: MODELS[f.model]?.color }} />
                    <strong>{MODELS[f.model]?.name ?? f.model}</strong> found it at #{f.rank}
                  </li>
                ))}
              </ul>
            ) : <p className="spotlight-note"><strong>No model found it this time</strong> - a miss is the usual case.</p>
          )}
        </div>
      </section>

      {compare.error && <p className="error">{compare.error}</p>}
      {!compare.data && !compare.error && <p className="loading">Loading recommendations…</p>}
      {rows.map((r) => (
        <ModelRow key={r.model} row={r} picks={picks} maxP={maxP} taste={tasteSet} />
      ))}

      <p className="chart-note">
        Already-watched movies are excluded from every row. p = the model's probability that this is the next movie;
        bars share one scale across rows. MATCH = shares a genre with the user's taste profile.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- pieces

function UserHeader({ user, taste, onShuffle, onPick, canShuffle }: {
  user: User; taste: [string, number][]; onShuffle: () => void; onPick: (id: number) => void; canShuffle: boolean;
}) {
  const [text, setText] = useState(String(user.user_id));
  useEffect(() => setText(String(user.user_id)), [user.user_id]);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const id = Number(text);
    if (Number.isInteger(id) && id > 0) onPick(id);
  };

  return (
    <section className="card user-head">
      <div className="avatar">#{user.user_id}</div>
      <div className="user-id">
        <div className="eyebrow muted">Now watching as</div>
        <h2>User #{user.user_id}</h2>
        <p>{user.n_ratings.toLocaleString()} ratings in MovieLens</p>
      </div>
      <div className="taste">
        <div className="eyebrow muted">Taste profile · last {user.history.length}</div>
        <div className="chips">
          {taste.map(([g, n]) => {
            const bg = genreColor(g);
            return (
              <span key={g} className="chip" style={{ background: bg, color: inkFor(bg) }}>
                {g} <span className="chip-n">{n}</span>
              </span>
            );
          })}
        </div>
      </div>
      <div className="user-actions">
        <button className="btn dark" onClick={onShuffle} disabled={!canShuffle}>⤮ Shuffle user</button>
        <form className="id-form" onSubmit={submit}>
          <input aria-label="User ID" inputMode="numeric" value={text} onChange={(e) => setText(e.target.value)} />
          <button className="btn" type="submit">Go</button>
        </form>
      </div>
    </section>
  );
}

function ModelRow({ row, picks, maxP, taste }: {
  row: Recommendations; picks: Map<number, number>; maxP: number; taste: Set<string>;
}) {
  const info = MODELS[row.model] ?? { name: row.model, tag: "", desc: "", color: "var(--accent)" };
  const matches = row.recommendations.filter((r) => r.movie.genres.some((g) => taste.has(g))).length;
  const hit = row.recommendations.find((r) => r.is_next_movie);

  return (
    <section>
      <div className="row-head">
        <h3>{info.name}</h3>
        <span className="row-tag" style={{ color: info.color }}>{info.tag}</span>
        <span className="row-desc">{info.desc}</span>
        <span className="row-stats">
          <span className={`pill ${hit ? "pill-next" : "pill-miss"}`}>{hit ? `✓ Found it at #${hit.rank}` : "Missed it"}</span>
          <span className="row-match">Genre match <strong>{matches}/{row.recommendations.length}</strong></span>
        </span>
      </div>
      {row.recommendations.length === 0 && <p className="empty">No recommendations for this user.</p>}
      <div className="poster-row">
        {row.recommendations.map((r) => {
          const n = picks.get(r.movie.movie_id) ?? 1;
          const match = r.movie.genres.some((g) => taste.has(g));
          return (
            <Poster key={r.movie.movie_id} movie={r.movie} highlight={r.is_next_movie} badges={<>
              {r.is_next_movie && <span className="pill pill-next">✓ Watched next</span>}
              {n > 1 && <span className="pill pill-count" title={`${n} models recommend it`}>×{n}</span>}
              {match && <span className="pill pill-match">Match</span>}
            </>}>
              <div className="foot-line">
                <span className="foot-rank">#{r.rank}</span>
                <span className="foot-p">p {fmtP(r.probability)}</span>
              </div>
              <div className="foot-track">
                <div className="foot-fill" style={{ width: `${(r.probability / maxP) * 100}%`, background: info.color }} />
              </div>
            </Poster>
          );
        })}
      </div>
    </section>
  );
}

