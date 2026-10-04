import { useEffect, useMemo, useState, type FormEvent } from "react";

import { useJson } from "../../lib/api";
import { Poster, WatchedNext, type Movie } from "./posters";

type User = { user_id: number; n_ratings: number; history: Movie[]; next_movie: Movie };
type Recommendation = { rank: number; movie: Movie; probability: number; is_next_movie: boolean };
type Recommendations = { user_id: number; model: string; hit: boolean; recommendations: Recommendation[] };
type ModelInfo = { model: string; hr_at_10: number };

export const API = "/api/genrec";
const HISTORY = 20;        // recent movies used for the "mostly ..." summary
const SHOWN = 9;           // of which the Watched row shows the latest few, after the movie they watched next

/** The model rows, in display order. The random / popular baselines stay in the API but aren't shown. */
const MODELS: Record<string, { name: string }> = {
  qwen: { name: "GenRec" },
  sasrec: { name: "SASRec" },
  "tfrs-sequential": { name: "TFRS" },
};
const ORDER = Object.keys(MODELS);
const rankOf = (m: string) => (ORDER.indexOf(m) + 1 || 99);
const nameOf = (m: string) => MODELS[m]?.name ?? m;

const pct = (p: number) => `${(p * 100).toFixed(p < 0.01 ? 2 : 1)}%`;

function pick<T>(xs: T[]): T {
  return xs[Math.floor(Math.random() * xs.length)];
}

/** The user's most frequent genres in their recent history. */
function topGenres(history: Movie[], n = 3): string[] {
  const counts = new Map<string, number>();
  history.forEach((m) => m.genres.forEach((g) => counts.set(g, (counts.get(g) ?? 0) + 1)));
  return [...counts].sort((a, b) => b[1] - a[1]).slice(0, n).map(([g]) => g);
}

// ---------------------------------------------------------------- the explorer

export default function Explorer({ userId, users, onPick }: {
  userId: number | null; users: number[] | null; onPick: (id: number) => void;
}) {
  // each model's test accuracy, fetched once: HR@10 = share of the 1,000 test users whose next movie is in its top 10
  const models = useJson<ModelInfo[]>(`${API}/models`);
  const hr = useMemo(() => new Map(models.data?.map((m) => [m.model, m.hr_at_10])), [models.data]);
  const shuffle = () => users?.length && onPick(pick(users.filter((u) => u !== userId)));
  if (userId === null) return <p className="loading">Picking a user…</p>;
  return <UserExplorer key={userId} userId={userId} hr={hr} onShuffle={shuffle} onPick={onPick} />;
}

function UserExplorer({ userId, hr, onShuffle, onPick }: {
  userId: number; hr: Map<string, number>; onShuffle: () => void; onPick: (id: number) => void;
}) {
  const user = useJson<User>(`${API}/users/${userId}?history=${HISTORY}`);
  const compare = useJson<Recommendations[]>(`${API}/users/${userId}/compare`);
  const rows = useMemo(
    () => (compare.data ? compare.data.filter((r) => r.model in MODELS).sort((a, b) => rankOf(a.model) - rankOf(b.model)) : []),
    [compare.data],
  );

  const head = <Header user={user.data} userId={userId} onShuffle={onShuffle} onPick={onPick} />;
  if (user.error) return <>{head}<p className="error">{user.error}</p></>;
  if (!user.data) return <>{head}<p className="loading">Loading…</p></>;

  const u = user.data;
  const found = rows.filter((r) => r.hit)
    .map((r) => `${nameOf(r.model)} #${r.recommendations.find((x) => x.is_next_movie)!.rank}`);

  return (
    <div className="explorer">
      {head}

      <section>
        <h2 className="row-title">Watched</h2>
        <div className="poster-row">
          <Poster movie={u.next_movie} highlight top={<WatchedNext />}
                  below={compare.data && (
                    <span className={found.length ? "found" : "not-found"}>
                      {found.length ? found.join(" · ") : "No model found it"}
                    </span>
                  )} />
          {[...u.history].reverse().slice(0, SHOWN).map((m) => <Poster key={m.movie_id} movie={m} />)}
        </div>
      </section>

      {compare.error && <p className="error">{compare.error}</p>}
      {rows.map((r) => (
        <section key={r.model}>
          <h2 className="row-title">
            {nameOf(r.model)}
            {hr.has(r.model) && (
              <span className="row-metric"
                    title={`HR@10: the actual next movie is in this model's top 10 for ${pct(hr.get(r.model)!)} of the 1,000 test users`}>
                {pct(hr.get(r.model)!)} in top 10
              </span>
            )}
          </h2>
          {r.recommendations.length === 0 && <p className="empty">No recommendations for this user.</p>}
          <div className="poster-row">
            {r.recommendations.map((x) => (
              <Poster key={x.movie.movie_id} movie={x.movie} highlight={x.is_next_movie}
                      top={<>
                        <span className="poster-p">{pct(x.probability)}</span>
                        {x.is_next_movie && <WatchedNext />}
                      </>} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function Header({ user, userId, onShuffle, onPick }: {
  user: User | null; userId: number; onShuffle: () => void; onPick: (id: number) => void;
}) {
  const [text, setText] = useState("");
  useEffect(() => setText(""), [userId]);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const id = Number(text);
    if (Number.isInteger(id) && id > 0) onPick(id);
  };

  return (
    <header className="user-head">
      <div>
        <h1>User {userId}</h1>
        {user && (
          <p>
            {user.n_ratings.toLocaleString()} ratings
            {user.history.length > 0 && <> · mostly {topGenres(user.history).join(", ")}</>}
          </p>
        )}
      </div>
      <div className="user-actions">
        <button className="link-btn" onClick={onShuffle}>Shuffle</button>
        <form className="id-form" onSubmit={submit}>
          <label htmlFor="user-id">ID</label>
          <input id="user-id" inputMode="numeric" value={text} onChange={(e) => setText(e.target.value)} />
          <button type="submit" aria-label="Show this user">→</button>
        </form>
      </div>
    </header>
  );
}
