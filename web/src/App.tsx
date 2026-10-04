import { Suspense, useEffect } from "react";
import { Link, Route, Routes, useLocation, useParams } from "react-router-dom";

import Home from "./pages/Home";
import NotFound from "./pages/NotFound";
import { PROJECTS, REPO_URL } from "./projects";

function ProjectRoute() {
  const { slug } = useParams();
  const project = PROJECTS.find((p) => p.slug === slug);
  useEffect(() => {
    document.title = project ? `${project.title.split(":")[0]} · Project Playground` : "Project Playground";
  }, [project]);
  if (!project) return <NotFound />;
  return (
    <Suspense fallback={<p className="loading">Loading…</p>}>
      <project.Page />
    </Suspense>
  );
}

export default function App() {
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);

  return (
    <>
      <header className="site-header">
        <div className="container">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" />
            Project Playground
          </Link>
          <nav className="nav">
            <Link to="/">Projects</Link>
            <a href="/api/docs">API</a>
            <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
          </nav>
        </div>
      </header>
      <main>
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/projects/:slug" element={<ProjectRoute />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </main>
      <footer className="site-footer">
        <div className="container">
          Built by <a href="https://github.com/mazprimrez" target="_blank" rel="noreferrer">mazprimrez</a> · code on{" "}
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
          <p className="credit">
            Movie posters from <a href="https://www.themoviedb.org/" target="_blank" rel="noreferrer">TMDB</a>. This
            product uses the TMDB API but is not endorsed or certified by TMDB.
          </p>
        </div>
      </footer>
    </>
  );
}
