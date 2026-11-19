"""In-memory dataset registry plus bundled sample datasets."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DATASET_ROWS = 200_000
MAX_DATASET_COLS = 200
MAX_CATEGORIES = 200
MAX_ONE_HOT_FEATURES = 5_000


@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    frame: pd.DataFrame

    @property
    def feature_columns(self) -> list[str]:
        return list(self.frame.columns)


def column_kind(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "categorical"
    if is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _preview(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.head(limit).to_dict(orient="records"):
        rows.append({key: _json_value(value) for key, value in record.items()})
    return rows


def _categories(frame: pd.DataFrame) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for column in frame.columns:
        if column_kind(frame[column]) != "categorical":
            continue
        values = [str(value) for value in frame[column].dropna().unique().tolist()]
        values = sorted(values)[:MAX_CATEGORIES]
        categories[column] = values
    return categories


def dataset_info(record: DatasetRecord, preview_limit: int = 5) -> dict[str, Any]:
    frame = record.frame
    missing = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
    }
    return {
        "dataset_id": record.dataset_id,
        "name": record.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "column_types": {column: str(frame[column].dtype) for column in frame.columns},
        "column_kinds": {column: column_kind(frame[column]) for column in frame.columns},
        "missing_values": missing,
        "categories": _categories(frame),
        "preview": _preview(frame, preview_limit),
    }


def preview_info(record: DatasetRecord, limit: int = 10) -> dict[str, Any]:
    info = dataset_info(record, preview_limit=limit)
    return {
        "dataset_id": info["dataset_id"],
        "name": info["name"],
        "rows": info["rows"],
        "columns": info["columns"],
        "column_names": info["column_names"],
        "column_kinds": info["column_kinds"],
        "missing_values": info["missing_values"],
        "preview": info["preview"],
    }


def register(frame: pd.DataFrame, name: str) -> DatasetRecord:
    dataset_id = uuid.uuid4().hex
    record = DatasetRecord(dataset_id=dataset_id, name=name, frame=frame)
    _DATASETS[dataset_id] = record
    return record


def get(dataset_id: str) -> DatasetRecord | None:
    return _DATASETS.get(dataset_id)


def validate_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if len(frame) > MAX_DATASET_ROWS:
        raise ValueError(f"Dataset has too many rows (max {MAX_DATASET_ROWS:,}).")
    if len(frame.columns) > MAX_DATASET_COLS:
        raise ValueError(f"Dataset has too many columns (max {MAX_DATASET_COLS}).")
    if any(str(column).strip() == "" for column in frame.columns):
        raise ValueError("Every column needs a name.")
    if len(set(map(str, frame.columns))) != len(frame.columns):
        raise ValueError("Column names must be unique.")


def detect_problem_type(series: pd.Series) -> str:
    """Mirror the UI's rule: bool / low-cardinality numeric / text => classification."""
    if is_bool_dtype(series):
        return "classification"
    if is_numeric_dtype(series):
        unique = int(series.nunique(dropna=True))
        if unique <= 10:
            return "classification"
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# Bundled sample datasets
# ---------------------------------------------------------------------------


def _iris() -> pd.DataFrame:
    from sklearn.datasets import load_iris

    bunch = load_iris()
    frame = pd.DataFrame(bunch.data, columns=[name.replace(" ", "_") for name in bunch.feature_names])
    frame["species"] = [str(bunch.target_names[value]) for value in bunch.target]
    return frame


def _wine() -> pd.DataFrame:
    from sklearn.datasets import load_wine

    bunch = load_wine()
    frame = pd.DataFrame(bunch.data, columns=[name.replace(" ", "_") for name in bunch.feature_names])
    frame["class"] = [str(bunch.target_names[value]) for value in bunch.target]
    return frame


def _breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    frame = pd.DataFrame(
        bunch.data, columns=[name.replace(" ", "_") for name in bunch.feature_names]
    )
    labels = {0: "malignant", 1: "benign"}
    frame["diagnosis"] = [labels[int(value)] for value in bunch.target]
    return frame


def _diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes()
    columns = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    frame = pd.DataFrame(bunch.data, columns=columns)
    frame["target"] = bunch.target.astype(float)
    return frame


SAMPLES: dict[str, dict[str, Any]] = {
    "iris": {
        "id": "iris",
        "name": "Iris",
        "task": "classification",
        "description": "Predict the iris species from four flower measurements.",
        "default_target": "species",
        "loader": _iris,
    },
    "wine": {
        "id": "wine",
        "name": "Wine",
        "task": "classification",
        "description": "Identify the wine cultivar from chemical analysis.",
        "default_target": "class",
        "loader": _wine,
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "name": "Breast Cancer",
        "task": "classification",
        "description": "Classify tumours as malignant or benign from cell measurements.",
        "default_target": "diagnosis",
        "loader": _breast_cancer,
    },
    "diabetes": {
        "id": "diabetes",
        "name": "Diabetes",
        "task": "regression",
        "description": "Predict disease progression from baseline measurements.",
        "default_target": "target",
        "loader": _diabetes,
    },
}


def sample_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": sample["id"],
            "name": sample["name"],
            "task": sample["task"],
            "description": sample["description"],
            "default_target": sample["default_target"],
        }
        for sample in SAMPLES.values()
    ]


def load_sample(sample_id: str) -> DatasetRecord:
    sample = SAMPLES.get(sample_id)
    if sample is None:
        raise KeyError(sample_id)
    frame = sample["loader"]()
    validate_frame(frame)
    return register(frame, sample["name"])


_DATASETS: dict[str, DatasetRecord] = {}
