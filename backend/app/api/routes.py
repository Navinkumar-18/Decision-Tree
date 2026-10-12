"""API routes for TreeViz AI.

In-memory storage (datasets + fitted models) is intentionally used for local
development: uploaded data and trained models are lost when the backend
restarts. No database or authentication is involved.
"""

import io
import json
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.trainer import ModelTrainer, detect_problem_type

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory stores (local development only)
# ---------------------------------------------------------------------------
datasets: Dict[str, Dict[str, Any]] = {}
models: Dict[str, Dict[str, Any]] = {}

# Configurable safety limits
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_ROWS = int(os.environ.get("MAX_DATASET_ROWS", 200_000))
MAX_COLS = int(os.environ.get("MAX_DATASET_COLS", 200))
PREVIEW_ROWS = 20

# Keys that hold fitted scikit-learn objects and must never be serialised.
_NON_SERIALIZABLE_KEYS = ("pipeline", "estimator")
# Keys excluded from the compact /summary payload (too large, fetched separately).
_SUMMARY_EXCLUDED_KEYS = _NON_SERIALIZABLE_KEYS + ("tree", "trees")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jsonable_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    """Convert a dataframe slice to plain JSON-safe records (NaN -> null)."""
    return json.loads(df.head(limit).to_json(orient="records"))


def _column_kinds(df: pd.DataFrame) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            kinds[str(col)] = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kinds[str(col)] = "numeric"
        else:
            kinds[str(col)] = "categorical"
    return kinds


def _categorical_options(df: pd.DataFrame, kinds: Dict[str, str]) -> Dict[str, List[str]]:
    """Distinct values for categorical columns (used to build input dropdowns)."""
    options: Dict[str, List[str]] = {}
    for col in df.columns:
        if kinds.get(str(col)) != "categorical":
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        nunique = int(series.nunique())
        if nunique > 50:
            continue
        try:
            values = sorted({str(v) for v in series.unique()})
        except TypeError:  # pragma: no cover - mixed types
            values = [str(v) for v in series.unique()]
        options[str(col)] = values[:50]
    return options


def _dataset_response(name: str, df: pd.DataFrame) -> "DatasetUploadResponse":
    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="The dataset has no columns.")
    kinds = _column_kinds(df)
    return DatasetUploadResponse(
        dataset_id=str(uuid.uuid4()),
        name=name,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        column_types={str(c): str(df[c].dtype) for c in df.columns},
        column_kinds=kinds,
        missing_values={str(c): int(df[c].isna().sum()) for c in df.columns},
        categories=_categorical_options(df, kinds),
        preview=_jsonable_records(df, PREVIEW_ROWS),
    )


def _read_csv(content: bytes, filename: str) -> pd.DataFrame:
    if not filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) can be uploaded.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The file is empty. Please upload a CSV file with data.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    try:
        df = pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file contains no data.")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read as CSV. Check that it is a valid, comma-separated file.",
        )
    if df.shape[0] == 0:
        raise HTTPException(status_code=400, detail="The CSV file has headers but no data rows.")
    if df.shape[1] < 2:
        raise HTTPException(
            status_code=400,
            detail="The dataset needs at least two columns (one target column and one feature column).",
        )
    if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
        raise HTTPException(
            status_code=400,
            detail=f"The dataset is too large for local processing (limit: {MAX_ROWS} rows, {MAX_COLS} columns).",
        )
    # Strip accidental whitespace / duplicate header names.
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        seen: Dict[str, int] = {}
        renamed = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                renamed.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                renamed.append(col)
        df.columns = renamed
    if df.empty or df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="The dataset could not be parsed into rows and columns.")
    return df


def _get_dataset(dataset_id: str) -> Dict[str, Any]:
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload it again.")
    return datasets[dataset_id]


def _get_model(model_id: str) -> Dict[str, Any]:
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found. Please generate it again.")
    return models[model_id]


def _serializable_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    """Model payload without fitted scikit-learn objects (keeps tree JSON)."""
    return {k: v for k, v in model.items() if k not in _NON_SERIALIZABLE_KEYS}


def _compact_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    """Compact configuration/metrics summary without the full tree JSON."""
    return {k: v for k, v in model.items() if k not in _SUMMARY_EXCLUDED_KEYS}


def _label(value: Any) -> str:
    if isinstance(value, float):
        from app.ml.tree_extractor import format_number

        return format_number(value)
    return str(value)


def _class_label(classes: List[str], raw: Any) -> str:
    """Map a raw estimator output onto the canonical class label."""
    for cls in classes:
        try:
            if float(cls) == float(raw):
                return cls
        except (TypeError, ValueError):
            continue
    return str(raw)


def _features_to_frame(model: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
    """Build a one-row dataframe matching the training columns and dtypes."""
    columns: List[str] = list(model.get("feature_columns") or [])
    if not columns:
        columns = [c for c in (model.get("pipeline").numeric_features or []) + (model.get("pipeline").categorical_features or [])]
    if not columns:
        raise HTTPException(status_code=400, detail="This model has no feature columns.")

    row: Dict[str, Any] = {}
    for col in columns:
        value = features.get(col)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            row[col] = np.nan
        else:
            row[col] = value
    frame = pd.DataFrame([row], columns=columns)

    pipeline = model.get("pipeline")
    numeric_cols = list(getattr(pipeline, "numeric_features", []) or [])
    categorical_cols = list(getattr(pipeline, "categorical_features", []) or [])
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in categorical_cols:
        if col in frame.columns:
            series = frame[col]
            frame[col] = series.where(series.notna(), other=np.nan).astype(object)
    return frame


def _model_payload(model: Dict[str, Any], prediction: Any, transformed: Any) -> Dict[str, Any]:
    """Common prediction payload shared by /predict and /decision-path."""
    estimator = model["estimator"]
    problem_type = model.get("problem_type", "classification")
    payload: Dict[str, Any] = {
        "model_type": model.get("type"),
        "problem_type": problem_type,
        "target_column": model.get("target_column"),
        "prediction": _label(prediction),
    }

    if problem_type == "classification":
        classes = [str(c) for c in getattr(estimator, "classes_", [])]
        payload["prediction"] = _class_label(classes, prediction) if classes else str(prediction)
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(transformed)[0]
                payload["probabilities"] = {
                    classes[i]: float(probs[i]) for i in range(len(classes))
                }
            except Exception:  # pragma: no cover
                payload["probabilities"] = None
        if model.get("type") == "random_forest":
            # Individual forest trees may expose float-typed classes, so map
            # every vote back onto the canonical class label.
            tree_preds: List[str] = []
            for sub in estimator.estimators_:
                if hasattr(sub, "predict_proba"):
                    idx = int(np.argmax(sub.predict_proba(transformed)[0]))
                    tree_preds.append(classes[idx] if idx < len(classes) else str(idx))
                else:  # pragma: no cover - defensive
                    tree_preds.append(_class_label(classes, sub.predict(transformed)[0]))
            payload["tree_predictions"] = tree_preds
            votes: Dict[str, int] = {}
            for p in tree_preds:
                votes[p] = votes.get(p, 0) + 1
            payload["tree_votes"] = votes
            payload["note"] = (
                "The final prediction combines the votes of every tree in the forest. "
                "A single tree on its own may disagree with the ensemble."
            )
    else:
        if model.get("type") == "random_forest":
            tree_preds = [
                float(sub.predict(transformed)[0]) for sub in estimator.estimators_
            ]
            payload["tree_predictions"] = tree_preds
            payload["note"] = (
                "The final prediction is the average of all individual tree outputs."
            )
    return payload


def _extracted_nodes(model: Dict[str, Any], tree_index: Optional[int]) -> List[Dict[str, Any]]:
    if model.get("type") == "random_forest":
        trees = model.get("trees") or []
        idx = 0 if tree_index is None else int(tree_index)
        if idx < 0 or idx >= len(trees):
            raise HTTPException(status_code=404, detail="Tree not found in this forest.")
        return trees[idx].get("nodes", [])
    tree = model.get("tree") or {}
    return tree.get("nodes", [])


def _decision_path_for(
    model: Dict[str, Any], transformed: np.ndarray, tree_index: Optional[int]
) -> Dict[str, Any]:
    estimator = model["estimator"]
    if model.get("type") == "random_forest":
        idx = 0 if tree_index is None else int(tree_index)
        trees = list(getattr(estimator, "estimators_", []))
        if idx < 0 or idx >= len(trees):
            raise HTTPException(status_code=404, detail="Tree not found in this forest.")
        tree_estimator = trees[idx]
    else:
        idx = None
        tree_estimator = estimator

    indicator = tree_estimator.decision_path(transformed)
    node_ids = sorted(int(n) for n in indicator.indices)
    if not node_ids:
        raise HTTPException(status_code=500, detail="Could not compute a decision path for this input.")

    nodes = {n["id"]: n for n in _extracted_nodes(model, idx)}
    rules: List[Dict[str, Any]] = []
    edges: List[List[int]] = []
    for parent, child in zip(node_ids, node_ids[1:]):
        info = nodes.get(parent, {})
        went_left = info.get("children_left") == child
        rule = info.get("left_rule") if went_left else info.get("right_rule")
        rules.append(
            {
                "node_id": parent,
                "feature": info.get("feature_display") or info.get("feature"),
                "condition": rule,
                "branch": (info.get("left_label") if went_left else info.get("right_label")),
                "went_left": bool(went_left),
            }
        )
        edges.append([parent, child])

    tree_prediction = tree_estimator.predict(transformed)[0]
    if model.get("problem_type") == "classification":
        classes = [str(c) for c in getattr(estimator, "classes_", [])]
        tree_label = _class_label(classes, tree_prediction) if classes else str(tree_prediction)
    else:
        tree_label = _label(float(tree_prediction) if np.isfinite(tree_prediction) else tree_prediction)

    return {
        "path": node_ids,
        "rules": rules,
        "edges": edges,
        "leaf_node_id": node_ids[-1],
        "tree_index": idx,
        "tree_prediction": tree_label,
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    column_kinds: Dict[str, str] = Field(default_factory=dict)
    missing_values: Dict[str, int]
    categories: Dict[str, List[str]] = Field(default_factory=dict)
    preview: List[Dict[str, Any]]


class ManualDataset(BaseModel):
    name: str
    columns: List[str]
    data: List[List[Any]]


class TrainRequest(BaseModel):
    dataset_id: str
    problem_type: str = "auto"
    algorithm: str = "decision_tree"
    target_column: str
    feature_columns: List[str] = []
    test_size: float = 0.2
    random_state: int = 42
    criterion: Optional[str] = None
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = 2
    min_samples_leaf: Optional[int] = 1
    max_leaf_nodes: Optional[int] = None
    n_estimators: Optional[int] = 10
    max_features: Optional[str] = "sqrt"


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class DecisionPathRequest(BaseModel):
    features: Dict[str, Any]
    tree_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@router.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read()
    df = _read_csv(content, file.filename or "")
    response = _dataset_response(file.filename or "dataset.csv", df)
    datasets[response.dataset_id] = {"name": response.name, "df": df}
    return response


@router.post("/datasets/manual", response_model=DatasetUploadResponse)
async def manual_dataset(payload: ManualDataset):
    if not payload.columns or not payload.data:
        raise HTTPException(status_code=400, detail="Invalid data")
    if len(payload.columns) < 2:
        raise HTTPException(
            status_code=400, detail="A dataset needs at least two columns."
        )
    try:
        df = pd.DataFrame(payload.data, columns=payload.columns)
    except Exception:
        raise HTTPException(status_code=400, detail="The provided rows do not match the columns.")
    if df.empty:
        raise HTTPException(status_code=400, detail="The dataset has no rows.")
    if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
        raise HTTPException(status_code=400, detail="The dataset is too large for local processing.")
    response = _dataset_response(payload.name or "manual.csv", df)
    datasets[response.dataset_id] = {"name": response.name, "df": df}
    return response


@router.get("/datasets")
async def list_samples():
    return {
        "samples": [
            {
                "id": "iris",
                "name": "Iris",
                "task": "classification",
                "description": "Classify iris flowers from petal and sepal measurements.",
                "default_target": "target",
            },
            {
                "id": "wine",
                "name": "Wine",
                "task": "classification",
                "description": "Recognise wine origin from chemical measurements.",
                "default_target": "target",
            },
            {
                "id": "breast_cancer",
                "name": "Breast Cancer Wisconsin",
                "task": "classification",
                "description": "Classify tumours as malignant or benign.",
                "default_target": "target",
            },
            {
                "id": "diabetes",
                "name": "Diabetes",
                "task": "regression",
                "description": "Predict disease progression one year after baseline.",
                "default_target": "target",
            },
        ]
    }


@router.post("/datasets/sample/{sample_id}", response_model=DatasetUploadResponse)
async def load_sample(sample_id: str):
    from sklearn.datasets import (
        load_breast_cancer,
        load_diabetes,
        load_iris,
        load_wine,
    )

    loaders = {
        "iris": (load_iris, "iris.csv"),
        "wine": (load_wine, "wine.csv"),
        "breast_cancer": (load_breast_cancer, "breast_cancer.csv"),
        "diabetes": (load_diabetes, "diabetes.csv"),
    }
    if sample_id not in loaders:
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    loader, filename = loaders[sample_id]
    data = loader(as_frame=True)
    df = data.frame.copy()
    response = _dataset_response(filename, df)
    datasets[response.dataset_id] = {"name": response.name, "df": df}
    return response


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@router.post("/models/train")
async def train_model(req: TrainRequest):
    if req.dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload it again.")
    df = datasets[req.dataset_id]["df"]
    config = req.model_dump()
    try:
        result = ModelTrainer.train_model(df, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail="The model could not be generated with this configuration. Try a different target column or setting.",
        )
    models[result["model_id"]] = result
    return _serializable_summary(result)


@router.get("/datasets/{dataset_id}/preview")
async def dataset_preview(dataset_id: str, limit: int = 10):
    stored = _get_dataset(dataset_id)
    df = stored["df"]
    limit = max(1, min(limit, PREVIEW_ROWS))
    kinds = _column_kinds(df)
    return {
        "dataset_id": dataset_id,
        "name": stored["name"],
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "column_kinds": kinds,
        "missing_values": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "preview": _jsonable_records(df, limit),
    }


@router.get("/models/{model_id}/summary")
async def model_summary(model_id: str):
    model = _get_model(model_id)
    summary = _compact_summary(model)
    estimator = model["estimator"]
    if model.get("type") == "random_forest":
        summary["trees_overview"] = [
            {
                "index": i,
                "depth": int(t.get("max_depth", 0)),
                "nodes": int(t.get("n_nodes", 0)),
                "leaves": int(t.get("n_leaves", 0)),
            }
            for i, t in enumerate(model.get("trees", []))
        ]
    else:
        tree = model.get("tree", {})
        summary["tree_overview"] = {
            "depth": int(tree.get("max_depth", 0)),
            "nodes": int(tree.get("n_nodes", 0)),
            "leaves": int(tree.get("n_leaves", 0)),
        }
    summary["estimator_class"] = type(estimator).__name__
    return summary


@router.get("/models/{model_id}/tree")
async def get_tree(model_id: str):
    model = _get_model(model_id)
    if model["type"] != "decision_tree":
        raise HTTPException(status_code=400, detail="This model is a random forest. Use the forest tree endpoints.")
    return model["tree"]


@router.get("/models/{model_id}/forest/trees")
async def forest_trees(model_id: str):
    model = _get_model(model_id)
    if model["type"] != "random_forest":
        raise HTTPException(status_code=400, detail="This model is not a random forest.")
    trees = model.get("trees", [])
    return {
        "n_estimators": len(trees),
        "trees": list(range(len(trees))),
        "overview": [
            {
                "index": i,
                "depth": int(t.get("max_depth", 0)),
                "nodes": int(t.get("n_nodes", 0)),
                "leaves": int(t.get("n_leaves", 0)),
            }
            for i, t in enumerate(trees)
        ],
    }


@router.get("/models/{model_id}/forest/trees/{tree_index}")
async def forest_tree(model_id: str, tree_index: int):
    model = _get_model(model_id)
    if model["type"] != "random_forest":
        raise HTTPException(status_code=400, detail="This model is not a random forest.")
    trees = model.get("trees", [])
    if tree_index < 0 or tree_index >= len(trees):
        raise HTTPException(status_code=404, detail="Tree not found in this forest.")
    return trees[tree_index]


@router.get("/models/{model_id}/feature-importance")
async def feature_importance(model_id: str):
    model = _get_model(model_id)
    return {
        "feature_names": model.get("feature_names", []),
        "feature_importances": model.get("feature_importances", []),
    }


# ---------------------------------------------------------------------------
# Predictions & decision paths (reuse the stored fitted model + pipeline)
# ---------------------------------------------------------------------------
@router.post("/models/{model_id}/predict")
async def predict(model_id: str, req: PredictRequest):
    model = _get_model(model_id)
    if not req.features:
        raise HTTPException(status_code=400, detail="Provide at least one feature value.")
    frame = _features_to_frame(model, req.features)
    transformed = model["pipeline"].transform(frame)
    prediction = model["estimator"].predict(transformed)[0]
    payload = _model_payload(model, prediction, transformed)
    return payload


@router.post("/models/{model_id}/decision-path")
async def decision_path(model_id: str, req: DecisionPathRequest):
    model = _get_model(model_id)
    if not req.features:
        raise HTTPException(status_code=400, detail="Provide at least one feature value.")
    frame = _features_to_frame(model, req.features)
    transformed = model["pipeline"].transform(frame)
    prediction = model["estimator"].predict(transformed)[0]
    payload = _model_payload(model, prediction, transformed)
    payload.update(_decision_path_for(model, transformed, req.tree_index))
    return payload


@router.get("/datasets/{dataset_id}/problem-type")
async def problem_type_hint(dataset_id: str, target_column: str):
    """Return the detected problem type for a target column (no training)."""
    stored = _get_dataset(dataset_id)
    df = stored["df"]
    if target_column not in df.columns:
        raise HTTPException(status_code=404, detail="Target column not found.")
    y = df[target_column].dropna()
    if y.empty:
        raise HTTPException(status_code=400, detail="The target column contains no valid values.")
    return {"problem_type": detect_problem_type(y)}
