"""Training, evaluation and inference for decision trees / random forests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..ml.tree_extractor import extract_tree, tree_path
from ..preprocessing.preprocessor import PreparedFeatures, build_preprocessor
from .datasets import DatasetRecord, detect_problem_type

CLASSIFICATION_CRITERIA = {"gini", "entropy", "log_loss"}
REGRESSION_CRITERIA = {"squared_error", "absolute_error", "friedman_mse"}


@dataclass
class ModelRecord:
    model_id: str
    estimator: Any
    prepared: PreparedFeatures
    algorithm: str
    problem_type: str
    detected_problem_type: str
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, Any]
    n_train: int
    n_test: int
    n_samples: int
    eval_on: str
    eval_note: str | None
    config: dict[str, Any] = field(default_factory=dict)
    tree_cache: dict[str, Any] | None = None

    @property
    def target_names(self) -> list[str] | None:
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            return None
        return [str(value) for value in classes]

    @property
    def feature_names(self) -> list[str]:
        return [meta.name for meta in self.prepared.metas]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "type": self.algorithm,
            "problem_type": self.problem_type,
            "problem_type_detected": self.detected_problem_type,
            "target_column": self.target_column,
            "criterion": self.config.get("criterion"),
            "metrics": self.metrics,
            "feature_importances": [float(value) for value in self.estimator.feature_importances_],
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_columns": self.feature_columns,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_samples": self.n_samples,
            "eval_on": self.eval_on,
            "eval_note": self.eval_note,
            "config": self.config,
        }
        if self.algorithm == "decision_tree":
            payload["tree"] = self.tree()
        else:
            payload["n_estimators"] = int(len(self.estimator.estimators_))
        return payload

    def tree(self) -> dict[str, Any]:
        if self.tree_cache is None:
            self.tree_cache = extract_tree(self.estimator, self.prepared, self.problem_type)
        return self.tree_cache

    def forest_tree(self, index: int) -> dict[str, Any]:
        estimators = getattr(self.estimator, "estimators_", [])
        if index < 0 or index >= len(estimators):
            raise IndexError(index)
        return extract_tree(estimators[index], self.prepared, self.problem_type)

    def transform_features(self, features: dict[str, Any]) -> np.ndarray:
        if not isinstance(features, dict) or not features:
            raise ValueError("Provide at least one feature value.")
        row = {column: features.get(column) for column in self.prepared.feature_columns}
        frame = pd.DataFrame([row])
        for column in self.prepared.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return self.prepared.transform(frame)


_MODELS: dict[str, ModelRecord] = {}


def get_model(model_id: str) -> ModelRecord | None:
    return _MODELS.get(model_id)


def _clean_target(series: pd.Series, problem_type: str) -> pd.Series:
    if problem_type == "classification":
        return series.dropna().astype(str)
    return pd.to_numeric(series, errors="coerce")


def _build_estimator(
    algorithm: str,
    problem_type: str,
    max_depth: int | None,
    min_samples_leaf: int,
    criterion: str | None,
    n_estimators: int,
    max_features: str | None,
) -> Any:
    classification = problem_type == "classification"
    if algorithm == "decision_tree":
        if classification:
            chosen = criterion if criterion in CLASSIFICATION_CRITERIA else "gini"
            return DecisionTreeClassifier(
                criterion=chosen,
                max_depth=max_depth,
                min_samples_leaf=max(1, min_samples_leaf),
                random_state=42,
            )
        chosen = criterion if criterion in REGRESSION_CRITERIA else "squared_error"
        return DecisionTreeRegressor(
            criterion=chosen,
            max_depth=max_depth,
            min_samples_leaf=max(1, min_samples_leaf),
            random_state=42,
        )

    if classification:
        chosen = criterion if criterion in CLASSIFICATION_CRITERIA else "gini"
        return RandomForestClassifier(
            n_estimators=max(1, min(100, n_estimators)),
            criterion=chosen,
            max_depth=max_depth,
            min_samples_leaf=max(1, min_samples_leaf),
            max_features=None if max_features in (None, "all") else max_features,
            random_state=42,
            n_jobs=-1,
        )
    chosen = criterion if criterion in REGRESSION_CRITERIA else "squared_error"
    return RandomForestRegressor(
        n_estimators=max(1, min(100, n_estimators)),
        criterion=chosen,
        max_depth=max_depth,
        min_samples_leaf=max(1, min_samples_leaf),
        max_features=None if max_features in (None, "all") else max_features,
        random_state=42,
        n_jobs=-1,
    )


def _evaluate(
    estimator: Any,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    problem_type: str,
) -> dict[str, Any]:
    predictions = estimator.predict(x_eval)
    if problem_type == "classification":
        labels = list(estimator.classes_)
        return {
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "precision": float(
                precision_score(
                    y_eval, predictions, average="weighted", zero_division=0, labels=labels
                )
            ),
            "recall": float(
                recall_score(
                    y_eval, predictions, average="weighted", zero_division=0, labels=labels
                )
            ),
            "f1_score": float(
                f1_score(
                    y_eval, predictions, average="weighted", zero_division=0, labels=labels
                )
            ),
            "confusion_matrix": confusion_matrix(y_eval, predictions, labels=labels).tolist(),
        }

    mse = float(mean_squared_error(y_eval, predictions))
    r2 = float(r2_score(y_eval, predictions)) if len(y_eval) >= 2 else None
    return {
        "mae": float(mean_absolute_error(y_eval, predictions)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2_score": r2,
    }


def train_model(
    dataset: DatasetRecord,
    target_column: str,
    algorithm: str,
    problem_type_choice: str,
    max_depth: int | None,
    min_samples_leaf: int,
    criterion: str | None,
    n_estimators: int,
    max_features: str | None,
) -> ModelRecord:
    if target_column not in dataset.frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    if algorithm not in ("decision_tree", "random_forest"):
        raise ValueError("Algorithm must be 'decision_tree' or 'random_forest'.")

    feature_columns = [column for column in dataset.frame.columns if column != target_column]
    if not feature_columns:
        raise ValueError("The dataset needs at least one feature column besides the target.")

    frame = dataset.frame
    mask = frame[target_column].notna()
    if not bool(mask.any()):
        raise ValueError(f"Target column '{target_column}' has no values to train on.")

    detected = detect_problem_type(frame[target_column])
    problem_type = problem_type_choice if problem_type_choice in ("classification", "regression") else detected

    target = _clean_target(frame.loc[mask, target_column], problem_type)
    valid = target.notna()
    target = target[valid]
    feature_frame = frame.loc[target.index, feature_columns]

    if len(target) < 2:
        raise ValueError("Not enough rows with a target value to train a model.")
    if problem_type == "classification":
        counts = target.value_counts()
        if len(counts) < 2:
            raise ValueError(
                "Classification needs at least two different target values. "
                "Choose regression or a different target column."
            )

    prepared = build_preprocessor(feature_frame, feature_columns)
    x_all = prepared.transform(feature_frame)
    y_all = target.to_numpy()

    if len(target) >= 10:
        stratify = None
        if problem_type == "classification" and int(target.value_counts().min()) >= 2:
            stratify = y_all
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x_all,
                y_all,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x_all, y_all, test_size=0.2, random_state=42
            )
        eval_on = "test"
        eval_note = None
    else:
        x_train, y_train = x_all, y_all
        x_test, y_test = x_all, y_all
        eval_on = "train"
        eval_note = "Fewer than 10 rows: the model is evaluated on the training data."

    estimator = _build_estimator(
        algorithm=algorithm,
        problem_type=problem_type,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        n_estimators=n_estimators,
        max_features=max_features,
    )
    estimator.fit(x_train, y_train)

    metrics = _evaluate(estimator, x_test, y_test, problem_type)

    config = {
        "algorithm": algorithm,
        "max_depth": max_depth,
        "min_samples_leaf": max(1, min_samples_leaf),
        "criterion": criterion,
        "n_estimators": int(n_estimators) if algorithm == "random_forest" else None,
        "max_features": max_features,
    }

    record = ModelRecord(
        model_id=uuid.uuid4().hex,
        estimator=estimator,
        prepared=prepared,
        algorithm=algorithm,
        problem_type=problem_type,
        detected_problem_type=detected,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=metrics,
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        n_samples=int(len(target)),
        eval_on=eval_on,
        eval_note=eval_note,
        config=config,
    )
    _MODELS[record.model_id] = record
    return record


def _format_prediction(value: Any, problem_type: str) -> str:
    if problem_type == "classification":
        return str(value)
    number = float(value)
    if not np.isfinite(number):
        return "?"
    return f"{number:.6g}"


def _forest_tree_prediction(record: ModelRecord, estimator: Any, encoded: np.ndarray) -> str:
    """Sub-estimators of a forest predict encoded class indices; map them back."""
    raw = estimator.predict(encoded)[0]
    if record.problem_type == "classification":
        classes = getattr(record.estimator, "classes_", None)
        if classes is not None:
            try:
                return str(classes[int(round(float(raw)))])
            except (IndexError, TypeError, ValueError):
                pass
        return str(raw)
    return _format_prediction(raw, record.problem_type)


def predict(record: ModelRecord, features: dict[str, Any]) -> dict[str, Any]:
    encoded = record.transform_features(features)
    predictions = record.estimator.predict(encoded)
    base: dict[str, Any] = {
        "model_type": record.algorithm,
        "problem_type": record.problem_type,
        "target_column": record.target_column,
        "prediction": _format_prediction(predictions[0], record.problem_type),
    }

    if record.problem_type == "classification":
        probabilities = record.estimator.predict_proba(encoded)[0]
        classes = [str(value) for value in record.estimator.classes_]
        base["probabilities"] = {
            label: float(probability) for label, probability in zip(classes, probabilities)
        }
    else:
        base["probabilities"] = None

    if record.algorithm == "random_forest":
        estimators = record.estimator.estimators_
        tree_predictions = [
            _forest_tree_prediction(record, estimator, encoded)
            for estimator in estimators
        ]
        base["tree_predictions"] = tree_predictions
        if record.problem_type == "classification":
            votes: dict[str, int] = {}
            for value in tree_predictions:
                votes[value] = votes.get(value, 0) + 1
            base["tree_votes"] = votes
            base["note"] = (
                "The ensemble prediction is the majority vote of the individual trees."
            )
    return base


def decision_path(
    record: ModelRecord,
    features: dict[str, Any],
    tree_index: int | None,
) -> dict[str, Any]:
    encoded = record.transform_features(features)
    payload = predict(record, features)

    index: int | None = None
    estimator = record.estimator
    if record.algorithm == "random_forest":
        estimators = record.estimator.estimators_
        index = 0 if tree_index is None else int(tree_index)
        if index < 0 or index >= len(estimators):
            raise ValueError(f"Tree index {index} is out of range.")
        estimator = estimators[index]
        tree_payload = extract_tree(estimator, record.prepared, record.problem_type)
        payload["tree_index"] = index
        payload["tree_prediction"] = _forest_tree_prediction(record, estimator, encoded)
    else:
        tree_payload = record.tree()

    path, edges, rules, leaf_id = tree_path(estimator, tree_payload, encoded)
    payload.update(
        {
            "path": path,
            "edges": edges,
            "rules": rules,
            "leaf_node_id": leaf_id,
        }
    )
    return payload
