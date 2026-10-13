#!/usr/bin/env bash
# Production start command (Render-compatible).
# - cd's to this file's directory so `app.main` resolves regardless of CWD
#   (works from the repo root or when the service root directory is `backend`).
# - Binds 0.0.0.0 as required by container platforms.
# - Honours $PORT when the hosting platform supplies one (Render sets PORT).
set -eu
cd "$(dirname "$0")"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
