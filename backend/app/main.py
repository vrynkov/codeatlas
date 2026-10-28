"""The FastAPI application. Run locally with:
    uvicorn app.main:app --reload
For local development, the React app runs separately via `npm run dev` (Vite) and talks to
this server across localhost using CORS. In Docker (Phase 8) the frontend is built into static
files and this same server serves them directly, so CORS won't even be needed in production.
"""
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .logging_setup import bind_request, get_logger, setup_logging
from .tracing_setup import setup_tracing

setup_logging()
log = get_logger(__name__)

from .api.routes import router  # noqa: E402  after logging is set up, so its log lines are JSON too

app = FastAPI(title="CodeAtlas API")
setup_tracing(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One `request_finished` log line per request, with its duration. Every other log line written
    while handling the request carries the same request_id, so they can be grouped together.
    The /ask stream's own request_id (made up by the frontend) is reused, so the Stop button's id,
    the logs and the LangSmith trace all match.
    Note: for the /ask stream, duration_ms here is only the time until streaming STARTED; the full
    time is in the `team_run_finished` line."""
    rid = request.query_params.get("request_id") or request.headers.get("x-request-id") or uuid.uuid4().hex
    bind_request(rid, request.headers.get("x-cloud-trace-context"))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", method=request.method, path=request.url.path)
        raise
    duration_ms = round((time.perf_counter() - start) * 1000)
    # Polled every second while a repo indexes - keep it out of the normal (INFO) logs.
    level = log.debug if request.url.path.endswith("/index-status") else log.info
    level("request_finished", method=request.method, path=request.url.path,
          status=response.status_code, duration_ms=duration_ms)
    response.headers["X-Request-ID"] = rid
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # the Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.exists():  # only present after `npm run build` (see Phase 8) - safe to skip until then
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
