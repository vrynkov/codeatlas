"""The FastAPI application. Run locally with:
    uvicorn app.main:app --reload
For local development, the React app runs separately via `npm run dev` (Vite) and talks to
this server across localhost using CORS. In Docker (Phase 8) the frontend is built into static
files and this same server serves them directly, so CORS won't even be needed in production.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes import router

app = FastAPI(title="CodeAtlas API")

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
