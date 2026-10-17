import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import { useJson } from "../../lib/api";

const API = "/api/eras";
const ARTICLE = "https://medium.com/@sparklingdust/beating-my-own-self-on-swift-alert-mastermind-game-in-guessing-taylor-swift-eras-tour-outfit-52b6535e0d8f";

type Entry = { entry: string; name: string; description: string; points_per_show: number; outfit_points: number; song_points: number };
type Summary = { shows: number; max_points: number; categories: string[]; entries: Entry[]; source: string };
type Show = { show: number; date: string; city: string; night: number; leg: string; points: Record<string, number> };
type Pick = { pick: string | null; probability: number | null; correct: boolean | null; points: number };
type Category = { category: string; max_points: number; actual: string; picks: Record<string, Pick> };
type ShowDetail = Show & { categories: Category[] };

/** The entries shown as columns, in this order (the API has a couple more baselines, listed in the summary). */
const COLUMNS = ["choice", "prophet_lags", "prophet"];

const fmtDate = (d: string) =>
  new Date(`${d}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
const showName = (s: Show) => `${s.city}${s.night > 1 ? ` · night ${s.night}` : ""}`;
const pts = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(1));

export default function ErasPage() {
  const [params, setParams] = useSearchParams();
  const summary = useJson<Summary>(`${API}/summary`);
  const shows = useJson<Show[]>(`${API}/shows`);
  const list = shows.data ?? [];
  const param = params.get("show");
  const current = param !== null ? Number(param) : null;

  // no show in the URL: start with the last show of the tour (the URL keeps it, so a view can be shared)
  useEffect(() => {
    if (current === null && list.length) setParams({ show: String(list[list.length - 1].show) }, { replace: true });
  }, [current, list, setParams]);

  const index = list.findIndex((s) => s.show === current);
  const go = (i: number) => list[i] && setParams({ show: String(list[i].show) });
  const random = () => go(Math.floor(Math.random() * list.length));
  const names = useMemo(() => new Map(summary.data?.entries.map((e) => [e.entry, e.name])), [summary.data]);

  return (
    <div className="eras">
      <header className="eras-head">
        <div>
          <p className="eyebrow muted">Eras Tour replay · Swift Alert Mastermind</p>
          <h1>{index >= 0 ? showName(list[index]) : "Loading…"}</h1>
          {index >= 0 && <p>{fmtDate(list[index].date)} · {list[index].leg} · show {index + 1} of {list.length} replayed</p>}
        </div>
        <div className="eras-controls">
          <button className="btn" onClick={() => go(index - 1)} disabled={index <= 0} aria-label="Previous show">←</button>
          <select value={current ?? ""} onChange={(e) => setParams({ show: e.target.value })} aria-label="Pick a show">
            {groupByLeg(list).map(([leg, ss]) => (
              <optgroup key={leg} label={leg}>
                {ss.map((s) => <option key={s.show} value={s.show}>{fmtDate(s.date)} · {showName(s)}</option>)}
              </optgroup>
            ))}
          </select>
          <button className="btn" onClick={() => go(index + 1)} disabled={index < 0 || index >= list.length - 1}
                  aria-label="Next show">→</button>
          <button className="link-btn" onClick={random} disabled={!list.length}>Random show</button>
        </div>
      </header>

      <p className="eras-intro">
        Every show of the tour, replayed: each method guesses the night's outfit versions and surprise songs from the
        earlier shows only, then gets scored like the game. Two improvements on the Prophet forecasts of my{" "}
        <a href={ARTICLE} target="_blank" rel="noreferrer">original Medium experiment</a>: a choice model, and
        Prophet given what the earlier shows say about each version.
      </p>

      {shows.error && <p className="error">{shows.error}</p>}
      {current !== null && summary.data && (
        <ShowView key={current} show={current} max={summary.data.max_points} names={names} />
      )}
      {summary.data && <Overall summary={summary.data} />}
    </div>
  );
}

function groupByLeg(shows: Show[]): [string, Show[]][] {
  const legs = new Map<string, Show[]>();
  shows.forEach((s) => legs.set(s.leg, [...(legs.get(s.leg) ?? []), s]));
  return [...legs];
}

// ---------------------------------------------------------------- one show

function ShowView({ show, max, names }: { show: number; max: number; names: Map<string, string> }) {
  const detail = useJson<ShowDetail>(`${API}/shows/${show}`);
  if (detail.error) return <p className="error">{detail.error}</p>;
  if (!detail.data) return <p className="loading">Loading…</p>;
  const d = detail.data;
  const columns = COLUMNS.filter((c) => c in d.points);
  const best = Math.max(...columns.map((c) => d.points[c]));

  return (
    <>
      <section className="scoreboard" aria-label="Points at this show">
        {columns.map((c) => (
          <div key={c} className={`score e-${c}`}>
            <span className="score-name">{names.get(c) ?? c}</span>
            <span className="score-value">
              <strong>{pts(d.points[c])}</strong> / {max}
              {d.points[c] === best && columns.length > 1 && <span className="score-best">best</span>}
            </span>
            <span className="score-bar"><span style={{ width: `${(100 * d.points[c]) / max}%` }} /></span>
          </div>
        ))}
      </section>

      <div className="table-wrap">
        <table className="eras-table">
          <thead>
            <tr>
              <th scope="col">Category</th>
              <th scope="col">What she wore</th>
              {columns.map((c) => <th key={c} scope="col" className={`e-${c}`}>{names.get(c) ?? c}</th>)}
            </tr>
          </thead>
          <tbody>
            {d.categories.map((cat) => (
              <tr key={cat.category}>
                <th scope="row">
                  {cat.category}
                  <small>{cat.max_points} pts</small>
                </th>
                <td className="actual">{cat.actual}</td>
                {columns.map((c) => <PickCell key={c} pick={cat.picks[c]} />)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-note">
        Each method only saw the shows before this one. Outfits are 4 points each; a surprise song guess is 2 points
        if that song was played on that instrument (2024 mashups count every song in them). The Lover guitar (4
        points) isn't in the data, so the maximum here is {max}.
      </p>
    </>
  );
}

function PickCell({ pick }: { pick?: Pick }) {
  if (!pick || pick.pick === null) return <td className="pick none">—</td>;
  return (
    <td className={`pick ${pick.correct ? "hit" : "miss"}`}>
      <span className="mark" aria-label={pick.correct ? "correct" : "wrong"}>{pick.correct ? "✓" : "✗"}</span>
      <span>
        {pick.pick}
        {pick.probability !== null && <small>{Math.round(pick.probability * 100)}% sure</small>}
      </span>
    </td>
  );
}

// ---------------------------------------------------------------- the whole tour

function Overall({ summary }: { summary: Summary }) {
  const top = Math.max(...summary.entries.map((e) => e.points_per_show));
  return (
    <section className="overall">
      <h2 className="section-title">Across all {summary.shows} replayed shows</h2>
      <p className="section-sub">Average points per show (out of {summary.max_points}), outfits + surprise songs.</p>
      <ul className="overall-list">
        {summary.entries.map((e) => (
          <li key={e.entry} className={`e-${e.entry}`}>
            <div className="row">
              <strong>{e.name}</strong>
              <span className="overall-points">
                {e.points_per_show.toFixed(1)}
                <small> = {e.outfit_points.toFixed(1)} outfits + {e.song_points.toFixed(2)} songs</small>
              </span>
            </div>
            <span className="score-bar"><span style={{ width: `${(100 * e.points_per_show) / top}%` }} /></span>
            <p>{e.description}</p>
          </li>
        ))}
      </ul>
      <p className="chart-note">
        Data: the fan-made <a href={summary.source} target="_blank" rel="noreferrer">Eras Tour outfit and surprise song
        tracker</a>. The first 10 shows are only used for learning.
      </p>
    </section>
  );
}
