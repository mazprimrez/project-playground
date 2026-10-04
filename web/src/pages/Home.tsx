import { Link } from "react-router-dom";

import { useJson, type ProjectStatus } from "../lib/api";
import { PROJECTS } from "../projects";

export default function Home() {
  const { data: statuses } = useJson<ProjectStatus[]>("/api/");

  return (
    <div className="stack">
      <section className="hero">
        <h1>Project Playground</h1>
        <p>
          Self-contained machine-learning experiments: each one trained, evaluated against honest baselines, and served
          here as a live demo you can try.
        </p>
      </section>

      <section>
        <h2 className="section-title">Projects</h2>
        <p className="section-sub">Pick one to try it.</p>
        <div className="project-grid">
          {PROJECTS.map((p) => {
            const status = statuses?.find((s) => s.name === p.slug)?.status;
            return (
              <Link key={p.slug} to={`/projects/${p.slug}`} className="card project-card">
                <div className="row">
                  <div className="tags">
                    {p.tags.map((t) => (
                      <span key={t} className="tag">{t}</span>
                    ))}
                  </div>
                  {status && <span className={`status ${status}`}>{status === "up" ? "Live" : "Offline"}</span>}
                </div>
                <h3>{p.title}</h3>
                <p>{p.tagline}</p>
                <div className="stat">
                  {p.stat.label}: <strong>{p.stat.value}</strong>
                </div>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
}
