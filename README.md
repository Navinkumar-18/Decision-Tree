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
│   ├── tests/                    # 50 pytest tests
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── run.sh
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

Optional `.env` (see `.env.example`):

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Checks

```bash
# backend — 50 tests
cd backend && .venv/bin/python -m pytest tests -q

# frontend
npm run lint         # oxlint (0 errors, 0 warnings)
npx tsc --noEmit     # types
npm run build        # production bundle
```

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
  this is a local development/demo setup, not a multi-user service.
- Uploaded CSVs are limited in size and dimensions; validation errors are
  returned as friendly `400` messages.
- Missing feature values at predict time are imputed with the training-time
  imputer; a completely empty `features` object is rejected with `400`.
