"""Playground API: one server for every project's API, each mounted under its own prefix (/genrec/...).

The projects stay self-contained - each keeps its own serving/app.py that runs and deploys on its own. This file only
mounts them; it has no project logic. A project that fails to import or start answers 503 instead of taking the
others down, and GET /health says why.

Add a project: one line in PROJECTS (prefix -> its FastAPI app file), and its serve dependencies in the Dockerfile.

Run:  uvicorn app:app --reload        (index at http://localhost:8000/, each project's docs at /<prefix>/docs)
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent
PROJECTS = {
    "genrec": ROOT / "genrec" / "serving" / "app.py",
}

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


app = FastAPI(title="Playground API", version="0.1.0", lifespan=lifespan,
              description="Every project's API, each under its own prefix - see GET / for the list.")
for p in projects:
    app.mount(f"/{p.name}", p)


@app.get("/")
def index():
    """The projects, with links to their docs."""
    return [{"name": p.name, "title": p.app.title if p.app else None,
             "description": p.app.description if p.app else None,
             "status": "up" if p.up else "down", "docs": f"/{p.name}/docs"}
            for p in projects]


@app.get("/health")
def health():
    status = {p.name: "up" if p.up else p.error for p in projects}
    return {"status": "ok" if all(p.up for p in projects) else "degraded", "projects": status}
