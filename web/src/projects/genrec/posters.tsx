import { useState, type ReactNode } from "react";

export type Movie = {
  movie_id: number; title: string; genres: string[]; year: number | null; poster_url?: string | null;
};

/** Poster art: a color per genre. Decorative - every card also prints its genres. */
const GENRE_COLORS: Record<string, string> = {
  Action: "#c8412f", Adventure: "#2e7d55", Animation: "#e8a93a", "Children's": "#f2cf63", Comedy: "#ee8a35",
  Crime: "#2b2f36", Documentary: "#2f7f86", Drama: "#7b4a3a", Fantasy: "#6a4fb3", "Film-Noir": "#1e1e20",
  Horror: "#5a1620", Musical: "#c2407a", Mystery: "#3b4a8c", Romance: "#d0546e", "Sci-Fi": "#2b4bb0",
  Thriller: "#3a3f47", War: "#5b6b3a", Western: "#a8763e",
};
const FALLBACK = "#4b4a46";

export const genreColor = (genre: string) => GENRE_COLORS[genre] ?? FALLBACK;

/** The genre that colors a poster: the first one that isn't Drama or Comedy (too common to tell movies apart). */
const posterGenre = (genres: string[]) => genres.find((g) => g !== "Drama" && g !== "Comedy") ?? genres[0] ?? "";

/** Dark ink on light posters (yellow, orange), white on the rest. */
export function inkFor(hex: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.35 ? "#16150f" : "#ffffff";
}

/** "The Matrix (1999)" -> "The Matrix" */
export const bareTitle = (title: string) => title.replace(/\s*\(\d{4}\)$/, "");

/** The big letter: the first character, skipping a leading article ("The Matrix" -> M). */
export const posterLetter = (title: string) => bareTitle(title).replace(/^(The|A|An)\s+/i, "").charAt(0).toUpperCase();

export function Poster({ movie, badges, children, highlight }: {
  movie: Movie; badges?: ReactNode; children?: ReactNode; highlight?: boolean;
}) {
  const [broken, setBroken] = useState(false);
  const image = movie.poster_url && !broken ? movie.poster_url : null;
  const bg = genreColor(posterGenre(movie.genres));
  const meta = [movie.year, ...movie.genres.slice(0, 2)].filter(Boolean).join(" · ");
  return (
    <div className={`poster${highlight ? " is-next" : ""}`}>
      {image ? (
        // the real poster (it shows the title itself); the title is repeated underneath for reading and search
        <div className="poster-art has-image" style={{ background: bg }}>
          <img src={image} alt={`${bareTitle(movie.title)} poster`} loading="lazy" onError={() => setBroken(true)} />
          {badges && <div className="poster-badges">{badges}</div>}
        </div>
      ) : (
        // no poster found: a drawn one - genre color, big initial, title
        <div className="poster-art" style={{ background: bg, color: inkFor(bg) }}>
          <span className="poster-letter" aria-hidden="true">{posterLetter(movie.title)}</span>
          {badges && <div className="poster-badges">{badges}</div>}
          <div className="poster-title">{bareTitle(movie.title)}</div>
          <div className="poster-meta">{meta}</div>
        </div>
      )}
      <div className="poster-foot">
        {image && (
          <div className="poster-caption">
            <div className="caption-title">{bareTitle(movie.title)}</div>
            <div className="caption-meta">{meta}</div>
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
