"""Project Playground server: the portfolio UI at / and every project's API under /api/<project>/.

The projects stay self-contained - each keeps its own serving/app.py that runs and deploys on its own. This file only
mounts them; it has no project logic. A project that fails to import or start answers 503 instead of taking the
others down, and GET /api/health says why.

The UI (web/) is a static build (`npm run build` -> web/dist); any path that isn't a file gets index.html, so the
UI's own routes (/projects/genrec) work on reload.

Add a project: one line in PROJECTS (prefix -> its FastAPI app file), its serve dependencies in the Dockerfile, and
its page in web/src/projects/.

Run:  uvicorn app:app --reload        (UI at http://localhost:8000/ once built, API index at /api, docs at /api/docs,
                                       each project's docs at /api/<project>/docs)
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
PROJECTS = {
    "genrec": ROOT / "genrec" / "serving" / "app.py",
}
WEB_DIST = Path(os.environ.get("PLAYGROUND_WEB_DIST", ROOT / "web" / "dist"))

log = logging.getLogger("playground")


def load_app(name: str, path: Path) -> FastAPI:
    """Import a project's app by file path: every project has a `serving` package, so `import serving.app` would clash."""
    module_name = f"playground_{name}_serving"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module      # registered before running, so pydantic can resolve the module's types
    spec.loader.exec_module(module)
    return module.app


class Project:
    """A mounted project: forwards to its app, or answers 503 if it didn't import or start."""

    def __init__(self, name: str, path: Path):
        self.name, self.app, self.error = name, None, None
        try:
            self.app = load_app(name, path)
        except Exception as e:
            self.error = f"import failed: {e!r}"
            log.exception("project %s: import failed", name)

    @property
    def up(self) -> bool:
        return self.app is not None and self.error is None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not self.up:
            response = JSONResponse({"detail": f"{self.name} is unavailable: {self.error}"}, status_code=503)
            return await response(scope, receive, send)
        await self.app(scope, receive, send)


class SinglePageApp(StaticFiles):
    """Static files, with index.html for page routes that aren't files (the UI's routes are handled in the browser).
    A missing file with an extension (an old /assets/*.js after a deploy) stays a 404 instead of returning HTML."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code != 404 or "." in Path(path).name:
                raise
            return await super().get_response("index.html", scope)


def ui_app(dist: Path):
    """The built UI, or a pointer to how to build it."""
    if (dist / "index.html").exists():
        return SinglePageApp(directory=dist, html=True)
    hint = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @hint.get("/{path:path}")
    def not_built(path: str):
        return JSONResponse({"detail": "UI not built: cd web && npm install && npm run build "
                                       "(or npm run dev for development). The API is at /api."}, status_code=404)
    return hint


projects = [Project(name, path) for name, path in PROJECTS.items()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Starlette doesn't run a mounted app's lifespan (where e.g. genrec loads its tables), so run each one here
    async with AsyncExitStack() as stack:
        for p in projects:
            if p.app is None:
                continue
            try:
                await stack.enter_async_context(p.app.router.lifespan_context(p.app))
                p.error = None
            except Exception as e:
                p.error = f"startup failed: {e!r}"
                log.exception("project %s: startup failed", p.name)
        yield


api = FastAPI(title="Project Playground API", version="0.2.0",
              description="Every project's API, each under its own prefix - see GET /api for the list.")
for p in projects:
    api.mount(f"/{p.name}", p)


@api.get("/")
def index():
    """The projects, with links to their docs."""
    return [{"name": p.name, "title": p.app.title if p.app else None,
             "description": p.app.description if p.app else None,
             "status": "up" if p.up else "down", "docs": f"/api/{p.name}/docs"}
            for p in projects]


@api.get("/health")
def health():
    status = {p.name: "up" if p.up else p.error for p in projects}
    return {"status": "ok" if all(p.up for p in projects) else "degraded", "projects": status}


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", api)
app.mount("/", ui_app(WEB_DIST))
