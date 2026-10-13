# TreeViz AI — Decision Tree Visualizer

Interactive, single-page visualizer for decision trees and random forests.
Upload a dataset (or pick a bundled sample), choose a target column, and the app
**trains a real scikit-learn model automatically** — then renders it as an
explorable tree diagram with node inspection, decision-path highlighting,
per-tree browsing for forests, and live predictions.

**No "Train Model" button anywhere:** every meaningful configuration change
(dataset, target, algorithm, hyper-parameters) triggers regeneration, with
debouncing, request de-duplication and cancellation of stale requests.

## Stack

| Layer     | Tech |
|-----------|------|
| Frontend  | React 19, TypeScript, Vite 8, Tailwind CSS 4, React Flow (`@xyflow/react`), lucide-react |
| Backend   | FastAPI, scikit-learn 1.5, pandas, numpy |
| Tests     | pytest (backend), oxlint + `tsc --noEmit` (frontend) |

## Features

- **Auto-generation** — model retraining is automatic, debounced (600 ms) and
  cancelled when the configuration changes again.
- **Datasets** — CSV upload (validated: size/dimension limits, friendly errors),
  4 bundled sample datasets (iris, wine, breast cancer, diabetes), or manual
  entry; preview with missing-value counts and per-column kinds.
- **Problem-type detection** — classification vs. regression inferred from the
  target (bool / low-cardinality numeric / text → classification), with manual
  override in Advanced Settings.
- **Single decision tree** — full `tree_` extraction: readable conditions
  (`age <= 37.5`, `contract = Month-to-month`), Yes/No branch labels, class
  distribution & sample counts per node, gini/impurity values.
- **Original units** — numeric thresholds are shown in the caller's units
  (StandardScaler statistics are inverted for display; preprocessing itself is
  untouched).
- **Random forest** — per-tree overview (size, depth, accuracy) and a tree
  selector; each tree renders with the same detail as a single tree; predictions
  include ensemble votes (`tree_votes`, `tree_predictions`).
- **Node inspector** — click any node: condition, branch rules, path from root,
  leaf statistics and prediction.
- **Decision-path highlighting** — submit a row of features and the exact
  root→leaf path lights up on the canvas, with per-split rules.
- **Metrics** — accuracy / F1 / precision / recall for classification,
  MAE / RMSE / R² for regression; feature importances.
- **React Flow canvas** — tidy tree layout, pan/zoom, minimap, controls.

## Project structure

```
decision_tree/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app + CORS
│   │   ├── api/routes.py         # all endpoints (upload, train, predict, …)
│   │   ├── services/trainer.py   # ModelTrainer, detect_problem_type, metadata
│   │   ├── ml/tree_extractor.py  # sklearn tree_ → serialisable node graph
│   │   └── preprocessing/preprocessor.py  # ColumnTransformer pipeline
│   ├── tests/                    # 63 pytest tests
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── run.sh                    # local dev (reload, port 8000)
│   ├── start.sh                  # production start (0.0.0.0, $PORT)
│   └── .python-version           # Python 3.12 pin
├── src/
│   ├── App.tsx                   # single-page orchestration (auto-generation)
│   ├── components/               # canvas, nodes, panels, upload, inspector…
│   ├── services/api.ts           # axios client (VITE_API_BASE_URL)
│   ├── utils/treeLayout.ts       # tidy layout → React Flow nodes/edges
│   └── types/index.ts
└── README.md
```

## Getting started

### 1. Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # runtime
.venv/bin/pip install -r requirements-dev.txt    # + pytest, httpx (tests)

# serve on http://127.0.0.1:8000
.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# (or: source .venv/bin/activate && ./run.sh)
```

Health check: `curl http://127.0.0.1:8000/api/health`

### 2. Frontend

```bash
npm install
npm run dev          # http://localhost:3000  (proxies /api → :8000)
```

Local development needs **no** `.env`: the axios client falls back to a
same-origin `/api` base, and Vite's dev proxy forwards it to
`http://localhost:8000` (see `server.proxy` in `vite.config.ts`). To point the
dev server somewhere else, copy `.env.example` to `.env` and set
`VITE_API_BASE_URL`. The variable is read at **build** time.

### 3. Checks

```bash
# backend — 63 tests
cd backend && .venv/bin/python -m pytest tests -q

# frontend
npm run lint         # oxlint (0 errors, 0 warnings)
npx tsc --noEmit     # types
npm run build        # production bundle
```

## Deployment (Vercel + Render)

The app is deployed as **two separate services**:

| | Service | Directory |
|---|---------|-----------|
| Frontend | Vercel (static site) | repository root |
| Backend | Render (web service) | `backend/` |

Nothing is deployed automatically — configure both services manually as below.

### Backend on Render

- **Root Directory:** `backend` (Render dashboard field; `start.sh` also works
  when run from the repository root, so you may leave Root Directory empty)
- **Runtime:** Python 3.12 (pinned in `backend/.python-version`)
- **Build Command:** `pip install -r requirements.txt`
  (from the repo root instead: `pip install -r backend/requirements.txt`)
- **Start Command:** `sh start.sh`
  (from the repo root instead: `sh backend/start.sh`)

`start.sh` runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

It binds `0.0.0.0` and uses Render's `$PORT`. Do **not** use `run.sh` on
Render — it is the local-dev script (`--reload`, fixed port 8000).

- **Health Check Path:** `/` (default) or `/api/health` — both return `200`.
- **Workers:** leave at 1 (the `start.sh` default). Extra workers each get
  their own memory and would not share datasets/models.
- Optional **Test Command:** `pip install -r requirements-dev.txt && python -m pytest tests -q`

#### Backend environment variables (Render → Environment)

| Variable | Required | Value |
|----------|----------|-------|
| `CORS_ORIGINS` | **Yes** | Comma-separated allowed browser origins, e.g. `https://your-app.vercel.app` (add preview URLs like `https://your-app-git-branch-user.vercel.app` if needed). Defaults to localhost dev origins only — never set it to `*`. |
| `MAX_UPLOAD_BYTES` | No | Max CSV upload size in bytes (default `10485760` = 10 MB) |
| `MAX_DATASET_ROWS` | No | Max dataset rows (default `200000`) |
| `MAX_DATASET_COLS` | No | Max dataset columns (default `200`) |

### Frontend on Vercel

- **Framework Preset:** Vite (auto-detected)
- **Root Directory:** `.` (repository root)
- **Install Command:** `npm install`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Node:** `^20.19.0 || >=22.12.0` (enforced via `engines` in `package.json`)

#### Frontend environment variables (Vercel → Settings → Environment Variables)

| Variable | When | Value |
|----------|------|-------|
| `VITE_API_BASE_URL` | **Build time — Required** | Origin of the Render backend, e.g. `https://treeviz-api.onrender.com` (no trailing slash). Vite inlines this into the bundle; **redeploy after changing it.** |

Without `VITE_API_BASE_URL` the frontend calls `/api` on its own origin, which
Vercel does not serve — every request would 404.

### Cross-service behaviour

The split deployment needs no extra configuration beyond the two variables:

- CSV upload (`multipart/form-data`), Decision Tree and Random Forest training,
  predictions and decision-path requests are plain `POST`s from the axios
  client to `${VITE_API_BASE_URL}/api/...`; the backend serves `/api/*` under
  the CORS origin allow-list.
- Browser preflights (`OPTIONS`, sent for JSON `POST`s) are handled by
  FastAPI's `CORSMiddleware` with `allow_methods`/`allow_headers = *`.
- The dataset/model IDs the frontend passes back are keys of the backend's
  **in-memory** stores — they are only valid on the instance that created them
  (see limitations below).

### Storage limitations (read before deploying)

Datasets and trained models live **only in process memory**
(`datasets` / `models` dicts in `backend/app/api/routes.py`). There is no
database and no filesystem persistence (uploads are parsed from bytes, never
written to disk). Consequences:

- **Restart ⇒ data loss.** Every Render redeploy/restart clears all datasets
  and models; users must upload again. The frontend treats a vanished ID as an
  error and lets the user start over.
- **Single instance only.** Two instances (or autoscaling/preview environments)
  do not share state — an ID created by instance A returns `404` on instance B.
  Keep the service at **one instance**.
- **Free-tier spin-down.** Render's free plan sleeps the service; the first
  request after idle is slow, and any state is gone after a restart. Expect
  re-uploads on a fresh session.
- Model training happens in-request (up to 120 s client timeout); very large
  datasets are capped by `MAX_DATASET_*` to keep requests within platform
  limits. Render also enforces its own request timeouts — keep CSVs modest
  on free-tier instances, which are slower to train on.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | service status |
| GET  | `/api/datasets` | bundled sample datasets |
| POST | `/api/datasets/upload` | upload CSV (validated) |
| POST | `/api/datasets/manual` | create dataset from columns+rows |
| GET  | `/api/datasets/sample/{id}` | load a sample dataset |
| GET  | `/api/datasets/{id}/preview` | rows, kinds, missing counts |
| GET  | `/api/datasets/{id}/problem-type` | detect task for a target column |
| POST | `/api/models/train` | train DT/RF (auto problem type) |
| GET  | `/api/models/{id}/summary` | model metadata + metrics |
| GET  | `/api/models/{id}/tree` | single-tree node graph |
| GET  | `/api/models/{id}/forest/trees` | per-tree overview |
| GET  | `/api/models/{id}/forest/trees/{i}` | structure of tree *i* |
| GET  | `/api/models/{id}/feature-importance` | importances |
| POST | `/api/models/{id}/predict` | predict a feature row |
| POST | `/api/models/{id}/decision-path` | path/rules for a feature row |

## Notes & limitations

- Datasets and trained models are kept **in memory** (restart clears them) —
  see [Storage limitations](#storage-limitations-read-before-deploying).
- Uploaded CSVs are limited in size and dimensions; validation errors are
  returned as friendly `400` messages.
- Missing feature values at predict time are imputed with the training-time
  imputer; a completely empty `features` object is rejected with `400`.
- CORS is an explicit origin allow-list (`CORS_ORIGINS`); it is never
  unrestricted (`*`) in production.
