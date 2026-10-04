import { useState, type ReactNode } from "react";

export type Movie = {
  movie_id: number; title: string; genres: string[]; year: number | null; poster_url?: string | null;
};

/** "The Matrix (1999)" -> "The Matrix" */
export const bareTitle = (title: string) => title.replace(/\s*\(\d{4}\)$/, "");

/** A movie card: the TMDB poster (or a plain card when there is none) with the title over a fade at the bottom. */
export function Poster({ movie, top, below, highlight }: {
  movie: Movie; top?: ReactNode; below?: ReactNode; highlight?: boolean;
}) {
  const [broken, setBroken] = useState(false);
  const image = movie.poster_url && !broken ? movie.poster_url : null;
  return (
    <div className={`poster${highlight ? " is-next" : ""}`}>
      <div className={`poster-art${image ? " has-image" : ""}`} title={movie.title}>
        {image && <img src={image} alt="" loading="lazy" onError={() => setBroken(true)} />}
        {top && <div className="poster-top">{top}</div>}
        <div className="poster-text">
          <div className="poster-title">{bareTitle(movie.title)}</div>
          <div className="poster-meta">{movie.genres.slice(0, 2).join(" · ")}</div>
        </div>
      </div>
      {below && <div className="poster-below">{below}</div>}
    </div>
  );
}

export function WatchedNext() {
  return (
    <span className="watched-next">
      <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
        <path d="M8 3C4.4 3 1.7 5.6 1 8c.7 2.4 3.4 5 7 5s6.3-2.6 7-5c-.7-2.4-3.4-5-7-5Zm0 8a3 3 0 1 1 0-6 3 3 0 0 1 0 6Zm0-1.6a1.4 1.4 0 1 0 0-2.8 1.4 1.4 0 0 0 0 2.8Z"
              fill="currentColor" />
      </svg>
      Watched next
    </span>
  );
}
