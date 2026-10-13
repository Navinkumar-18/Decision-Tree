import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes

app = FastAPI(title="TreeViz AI API", version="1.0.0")

# Local development origins only. Override with CORS_ORIGINS (comma separated).
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
] or _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "title": "TreeViz AI API",
        "storage": "in-memory (local development only - data is lost on restart)",
    }


@app.get("/")
async def root():
    """Root route: Render's default health check expects a 2xx from `/`."""
    return {
        "status": "healthy",
        "title": "TreeViz AI API",
        "docs": "/docs",
        "health": "/api/health",
    }
