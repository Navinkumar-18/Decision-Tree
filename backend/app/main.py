"""FastAPI application entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def _allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="TreeViz AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}
