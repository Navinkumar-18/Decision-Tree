"""HTTP API routes."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from ..services.datasets import (
    MAX_UPLOAD_BYTES,
    detect_problem_type,
    dataset_info,
    get,
    load_sample,
    preview_info,
    register,
    sample_catalog,
    validate_frame,
)
from ..services.trainer import decision_path, get_model, predict, train_model

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class TrainBody(BaseModel):
    dataset_id: str
    target_column: str
    algorithm: str = "decision_tree"
    problem_type: str = "auto"
    max_depth: int | None = Field(default=4, ge=1, le=64)
    min_samples_leaf: int = Field(default=1, ge=1, le=100)
    criterion: str | None = None
    n_estimators: int = Field(default=10, ge=1, le=100)
    max_features: str | None = "sqrt"


class FeaturesBody(BaseModel):
    features: dict[str, Any]


class DecisionPathBody(BaseModel):
    features: dict[str, Any]
    tree_index: int | None = None


class ManualDatasetBody(BaseModel):
    name: str = "Manual dataset"
    columns: dict[str, list[Any]]


# ---------------------------------------------------------------------------
# Health / datasets
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    return {"samples": sample_catalog()}


@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"The file is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    if not payload.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="The file must be a UTF-8 encoded CSV.") from None
    except Exception as exc:  # pragma: no cover - pandas raises many parse errors
        raise HTTPException(status_code=400, detail=f"Could not parse the CSV: {exc}") from exc

    try:
        validate_frame(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = register(frame, file.filename or "Uploaded dataset")
    return dataset_info(record)


@router.post("/datasets/manual")
def create_manual_dataset(body: ManualDatasetBody) -> dict[str, Any]:
    if not body.columns:
        raise HTTPException(status_code=400, detail="Provide at least one column.")
    lengths = {len(values) for values in body.columns.values()}
    if len(lengths) != 1:
        raise HTTPException(status_code=400, detail="Every column must have the same length.")
    frame = pd.DataFrame(body.columns)
    try:
        validate_frame(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    record = register(frame, body.name)
    return dataset_info(record)


@router.post("/datasets/sample/{sample_id}")
def create_sample_dataset(sample_id: str) -> dict[str, Any]:
    try:
        record = load_sample(sample_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' was not found.") from None
    return dataset_info(record)


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    record = get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found or expired.")
    return preview_info(record, limit)


@router.get("/datasets/{dataset_id}/problem-type")
def problem_type(dataset_id: str, target_column: str = Query(...)) -> dict[str, str]:
    record = get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found or expired.")
    if target_column not in record.frame.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_column}' was not found.")
    return {"problem_type": detect_problem_type(record.frame[target_column])}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.post("/models/train")
def train(body: TrainBody) -> dict[str, Any]:
    dataset = get(body.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found or expired.")
    try:
        record = train_model(
            dataset=dataset,
            target_column=body.target_column,
            algorithm=body.algorithm,
            problem_type_choice=body.problem_type,
            max_depth=body.max_depth,
            min_samples_leaf=body.min_samples_leaf,
            criterion=body.criterion,
            n_estimators=body.n_estimators,
            max_features=body.max_features,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.to_payload()


def _model_or_404(model_id: str):
    record = get_model(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found or expired.")
    return record


@router.get("/models/{model_id}/summary")
def model_summary(model_id: str) -> dict[str, Any]:
    return _model_or_404(model_id).to_payload()


@router.get("/models/{model_id}/tree")
def model_tree(model_id: str) -> dict[str, Any]:
    record = _model_or_404(model_id)
    if record.algorithm == "decision_tree":
        return record.tree()
    return record.forest_tree(0)


@router.get("/models/{model_id}/forest/trees")
def forest_trees(model_id: str) -> dict[str, Any]:
    record = _model_or_404(model_id)
    if record.algorithm != "random_forest":
        raise HTTPException(status_code=400, detail="This model is not a random forest.")
    estimators = record.estimator.estimators_
    overview = []
    for index, estimator in enumerate(estimators):
        tree = estimator.tree_
        overview.append(
            {
                "index": index,
                "depth": int(tree.max_depth),
                "nodes": int(tree.node_count),
                "leaves": int(tree.n_leaves),
            }
        )
    return {
        "n_estimators": len(estimators),
        "trees": list(range(len(estimators))),
        "overview": overview,
    }


@router.get("/models/{model_id}/forest/trees/{tree_index}")
def forest_tree(model_id: str, tree_index: int) -> dict[str, Any]:
    record = _model_or_404(model_id)
    if record.algorithm != "random_forest":
        raise HTTPException(status_code=400, detail="This model is not a random forest.")
    try:
        return record.forest_tree(tree_index)
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Tree {tree_index} was not found.") from None


@router.get("/models/{model_id}/feature-importance")
def feature_importance(model_id: str) -> dict[str, Any]:
    record = _model_or_404(model_id)
    return {
        "feature_names": record.feature_names,
        "importances": [float(value) for value in record.estimator.feature_importances_],
    }


@router.post("/models/{model_id}/predict")
def predict_route(model_id: str, body: FeaturesBody) -> dict[str, Any]:
    record = _model_or_404(model_id)
    try:
        return predict(record, body.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{model_id}/decision-path")
def decision_path_route(model_id: str, body: DecisionPathBody) -> dict[str, Any]:
    record = _model_or_404(model_id)
    try:
        return decision_path(record, body.features, body.tree_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
