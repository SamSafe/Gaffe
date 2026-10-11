"""Local API and bundled frontend. No credentials or arbitrary commands exposed."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fpl_bot.web.jobs import ACTIVE, BusyError, JobManager
from fpl_bot.web.schemas import JobRequest, PlanRequest, Preferences
from fpl_bot.web.service import dashboard, evaluate_plan
from fpl_bot.web.store import Store

ASSETS = Path(__file__).parent / "static"


def create_app(store: Store | None = None) -> FastAPI:
    store = store or Store()
    token = secrets.token_urlsafe(32)
    manager = JobManager(store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="Gaffe", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.jobs = manager
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "Use this app's own browser window."}, status_code=403
                )
            supplied = request.headers.get("x-gaffe-token", "")
            if not secrets.compare_digest(supplied, token):
                return JSONResponse(
                    {"detail": "Reload the app before making changes."}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/dashboard")
    def get_dashboard():
        return {**dashboard(store), "csrf_token": token, "jobs": manager.list()}

    @app.get("/api/preferences")
    def get_preferences():
        return store.preferences()

    @app.put("/api/preferences")
    def put_preferences(prefs: Preferences):
        if any(j["status"] in ACTIVE for j in manager.list()):
            raise HTTPException(409, "Wait for the active job before changing settings.")
        store.save_preferences(prefs)
        return prefs

    @app.post("/api/plan")
    def plan(request: PlanRequest):
        try:
            return evaluate_plan(store, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/jobs")
    def jobs():
        return manager.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        data = manager.get(job_id)
        if not data:
            raise HTTPException(404, "Job not found.")
        return data

    @app.post("/api/jobs", status_code=202)
    def start_job(request: JobRequest):
        try:
            return manager.start(request, store.preferences())
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
    return app
