import { useEffect, useRef, useState } from "react";

import { getJson } from "../../lib/api";
import { API } from "./Explorer";
import { Poster, type Movie } from "./posters";

/** The live model service (Cloud Run `genrec-model`); VITE_MODEL_URL overrides it in development. */
const MODEL_URL = import.meta.env.VITE_MODEL_URL ?? "https://genrec-model-114282263198.asia-southeast2.run.app";
const MAX_PICKS = 20;            // the model reads the last 20 movies

type Reason = { movie_id: number; title: string; drop: number };
type Pick = { rank: number; movie_id: number; title: string; genres: string[]; year: number | null;
              poster_url: string | null; probability: number; because: Reason[] | null };
type Result = { recommendations: Pick[]; seconds: number; num_beams: number };

const pct = (p: number) => `${(p * 100).toFixed(p < 0.01 ? 2 : 1)}%`;

export default function TryYourOwn() {
  const [picks, setPicks] = useState<Movie[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [explain, setExplain] = useState(false);

  // count the seconds while the model works
  useEffect(() => {
    if (!running) return;
    const t0 = Date.now();
    const timer = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [running]);

  const add = (m: Movie) => setPicks((p) => (p.some((x) => x.movie_id === m.movie_id) || p.length >= MAX_PICKS ? p : [...p, m]));
  const remove = (id: number) => setPicks((p) => p.filter((m) => m.movie_id !== id));

  const recommend = async () => {
    setRunning(true); setElapsed(0); setError(null); setResult(null);
    try {
      const res = await fetch(`${MODEL_URL}/recommend`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ movie_ids: picks.map((m) => m.movie_id), k: 10, explain }),
        signal: AbortSignal.timeout(10 * 60 * 1000),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(res.status === 503 ? "The model is still waking up - try again in a minute."
                                           : body?.detail ?? `${res.status} ${res.statusText}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="explorer">
      <section>
        <h2 className="row-title">Your movies <span className="row-metric">in the order you watched them, up to {MAX_PICKS}</span></h2>
        <MovieSearch onPick={add} exclude={new Set(picks.map((m) => m.movie_id))} disabled={picks.length >= MAX_PICKS} />
        {picks.length > 0 && (
          <div className="poster-row picked">
            {picks.map((m) => (
              <Poster key={m.movie_id} movie={m}
                      top={<button className="remove" onClick={() => remove(m.movie_id)} aria-label={`Remove ${m.title}`}>×</button>} />
            ))}
          </div>
        )}
      </section>

      <section className="try-run">
        <button className="btn dark" onClick={recommend} disabled={!picks.length || running}>
          {running ? `Thinking… ${elapsed}s` : "Recommend"}
        </button>
        <label className="try-explain">
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} disabled={running} />
          Explain why
        </label>
        <span className="try-note">
          {running
            ? explain
              ? "Recommending, then testing which of your last 10 movies each pick depends on: this can take a few minutes."
              : "The model runs on a CPU: this takes up to a minute, longer if it has to wake up first."
            : explain
              ? "Each pick gets the movies it depends on most - measured by removing them one at a time. Takes a few minutes."
              : "GenRec (Qwen2.5-0.5B) reads your list and writes the titles it expects you to watch next."}
        </span>
      </section>

      {error && <p className="error">{error}</p>}
      {result && (
        <section>
          <h2 className="row-title">GenRec recommends <span className="row-metric">answered in {result.seconds}s</span></h2>
          <div className="poster-row">
            {result.recommendations.map((r) => (
              <Poster key={r.movie_id}
                      movie={{ movie_id: r.movie_id, title: r.title, genres: r.genres, year: r.year, poster_url: r.poster_url }}
                      top={<span className="poster-p">{pct(r.probability)}</span>}
                      below={r.because && (r.because.length
                        ? <span className="because">Because of {r.because.map((b, i) => (
                            <span key={b.movie_id}>{i > 0 && ", "}<strong>{b.title.replace(/ \(\d{4}\)$/, "")}</strong> −{Math.round(b.drop * 100)}%</span>))}</span>
                        : <span className="because muted">No single movie stands out</span>)} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function MovieSearch({ onPick, exclude, disabled }: {
  onPick: (m: Movie) => void; exclude: Set<number>; disabled: boolean;
}) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Movie[]>([]);
  const box = useRef<HTMLDivElement>(null);

  // search the catalogue as you type (debounced); the site API answers instantly, the model isn't involved
  useEffect(() => {
    const text = q.trim();
    if (!text) { setHits([]); return; }
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      getJson<Movie[]>(`${API}/movies?q=${encodeURIComponent(text)}&limit=8`, ctrl.signal).then(setHits).catch(() => {});
    }, 250);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [q]);

  // close the list when clicking elsewhere
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setHits([]); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const shown = hits.filter((m) => !exclude.has(m.movie_id));
  return (
    <div className="search" ref={box}>
      <input type="search" placeholder={disabled ? "That's the maximum" : "Search a movie from 2010-2023…"}
             value={q} onChange={(e) => setQ(e.target.value)} disabled={disabled} aria-label="Search movies" />
      {shown.length > 0 && (
        <ul className="search-results" role="listbox">
          {shown.map((m) => (
            <li key={m.movie_id}>
              <button onClick={() => { onPick(m); setQ(""); setHits([]); }}>
                {m.poster_url ? <img src={m.poster_url} alt="" /> : <span className="thumb" />}
                <span>{m.title}<small>{m.genres.slice(0, 3).join(" · ")}</small></span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
