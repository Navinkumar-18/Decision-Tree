"""Model training service.

Kept as a single ``ModelTrainer.train_model`` entry point so the API layer stays
thin. The module trains real scikit-learn estimators, evaluates them on a held
out test split and extracts the actual fitted tree structure for visualisation.

Improvements over the original version:
- problem type detection (classification vs regression) with an explicit override
- the preprocessing pipeline is fitted on the training split only
- friendly validation errors (no raw stack traces for expected mistakes)
- small datasets fall back to training-set evaluation with a clear note
- the fitted pipeline and estimator are stored so later prediction /
  decision-path requests reuse the exact same model
"""

from typing import Any, Dict, List, Optional, Tuple

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
import uuid

from app.ml.tree_extractor import TreeExtractor
from app.preprocessing.preprocessor import DataPreprocessor

RANDOM_STATE = 42
MAX_TREES = 100
MAX_DEPTH_LIMIT = 64

CLASSIFICATION_CRITERIA = ("gini", "entropy", "log_loss")
REGRESSION_CRITERIA = ("squared_error", "friedman_mse", "absolute_error", "poisson")
MAX_FEATURES_OPTIONS = ("sqrt", "log2", "all")

# Legacy scikit-learn aliases accepted for backwards compatibility.
CRITERION_ALIASES = {"mse": "squared_error", "rmse": "squared_error", "mae": "absolute_error"}


def detect_problem_type(y: pd.Series) -> str:
    """Infer whether the target column looks like classification or regression.

    Never decided by dtype alone. The explicit ``problem_type`` request always
    wins (see ``ModelTrainer.train_model``); this helper only powers ``auto``:

    - boolean and non-numeric columns are class labels;
    - numeric columns with very few distinct values are class labels too
      (0/1 flags, 1-5 ratings, small integer codes) even though they are numeric;
    - numeric columns with many distinct values (continuous measurements such as
      ``Study_Hours``) stay regression targets.

    Ambiguous columns (e.g. a continuous-looking column with few observed
    values) are resolved with the explicit problem-type setting in Advanced
    Settings rather than a silent guess.
    """
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        n_unique = int(y.nunique(dropna=True))
        n = int(y.notna().sum())
        if n_unique <= 2:
            return "classification"
        ratio = n_unique / max(n, 1)
        if n_unique <= 20 and ratio <= 0.5:
            return "classification"
        return "regression"
    return "classification"


def resolve_criterion(raw_criterion: Any, problem_type: str) -> str:
    """Map a requested criterion onto a value that is valid for ``problem_type``.

    The UI sends its criterion independently of the task (the default is
    ``gini`` while ``problem_type`` is ``auto``), so a criterion that is valid
    for the *other* task is mapped onto this task's default instead of being
    handed to scikit-learn, which rejects it. Only values that are not a
    criterion of either task are reported as errors.
    """
    allowed = CLASSIFICATION_CRITERIA if problem_type == "classification" else REGRESSION_CRITERIA
    if raw_criterion is None or str(raw_criterion).strip() in ("", "None", "none"):
        return allowed[0]
    value = str(raw_criterion).strip().lower()
    value = CRITERION_ALIASES.get(value, value)
    if value in allowed:
        return value
    if value in CLASSIFICATION_CRITERIA or value in REGRESSION_CRITERIA:
        # Valid, but for the other task -> use this task's default instead.
        return allowed[0]
    raise ValueError(f"Criterion must be one of: {', '.join(allowed)}.")


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _optional_int(value: Any, name: str) -> Optional[int]:
    if value is None or value == "" or value == "None":
        return None
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a whole number.")
    if iv < 1:
        raise ValueError(f"'{name}' must be at least 1.")
    return iv


def _required_int(value: Any, name: str, minimum: int = 1) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a whole number.")
    if iv < minimum:
        raise ValueError(f"'{name}' must be at least {minimum}.")
    return iv


def _build_feature_meta(
    preprocessor: DataPreprocessor, feature_names: List[str]
) -> Dict[int, Dict[str, Any]]:
    """Map every transformed feature back to its source column.

    For numeric columns the scaler statistics are attached so thresholds can be
    displayed in original units. For one-hot encoded columns the source column
    and the encoded category are attached so rules read like
    ``contract_type = Month-to-month`` instead of ``cat__contract_type_...``.
    """
    meta: Dict[int, Dict[str, Any]] = {}
    numeric_features = list(getattr(preprocessor, "numeric_features", []) or [])
    categorical_features = list(getattr(preprocessor, "categorical_features", []) or [])

    means: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    try:
        transformer = preprocessor.column_transformer.named_transformers_.get("num")
        scaler = transformer.named_steps.get("scaler") if transformer is not None else None
        if scaler is not None and hasattr(scaler, "mean_"):
            for i, col in enumerate(numeric_features):
                means[str(col)] = float(scaler.mean_[i])
                scales[str(col)] = float(scaler.scale_[i])
    except Exception:  # pragma: no cover - defensive, scaler optional
        pass

    index = 0
    for col in numeric_features:
        if index < len(feature_names):
            meta[index] = {
                "source": str(col),
                "kind": "numeric",
                "mean": means.get(str(col)),
                "scale": scales.get(str(col)),
            }
        index += 1

    # One-hot block sizes come straight from the fitted encoder, so columns
    # whose names contain underscores are still attributed correctly.
    category_counts: List[int] = []
    try:
        encoder = (
            preprocessor.column_transformer.named_transformers_["cat"].named_steps["encoder"]
        )
        category_counts = [len(c) for c in encoder.categories_]
    except Exception:  # pragma: no cover - defensive
        category_counts = []

    for position, col in enumerate(categorical_features):
        if position < len(category_counts):
            count = category_counts[position]
        else:  # pragma: no cover - fallback when encoder metadata unavailable
            prefix = f"cat__{col}_"
            count = 0
            probe = index
            while probe < len(feature_names) and str(feature_names[probe]).startswith(prefix):
                count += 1
                probe += 1
        for k in range(count):
            if index >= len(feature_names):
                break
            name = str(feature_names[index])
            prefix = f"cat__{col}_"
            category = name[len(prefix):] if name.startswith(prefix) else name
            meta[index] = {
                "source": str(col),
                "kind": "categorical",
                "category": category,
            }
            index += 1

    return meta


class ModelTrainer:
    @staticmethod
    def train_model(dataset_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        target_column = config.get("target_column")
        if not target_column or not isinstance(target_column, str):
            raise ValueError("Please choose a target column to predict.")
        if target_column not in dataset_df.columns:
            raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

        problem_type = str(config.get("problem_type") or "auto").lower()
        if problem_type not in ("auto", "classification", "regression"):
            raise ValueError("Problem type must be 'classification', 'regression' or 'auto'.")

        algorithm = config.get("algorithm") or "decision_tree"
        if algorithm not in ("decision_tree", "random_forest"):
            raise ValueError("Model type must be 'decision_tree' or 'random_forest'.")

        feature_columns = [
            c
            for c in (config.get("feature_columns") or [])
            if c in dataset_df.columns and c != target_column
        ]
        if feature_columns:
            X = dataset_df[feature_columns].copy()
        else:
            X = dataset_df.drop(columns=[target_column]).copy()
        y = dataset_df[target_column].copy()

        if X.shape[1] == 0:
            raise ValueError("The dataset needs at least one feature column besides the target.")

        # Drop rows where the target is missing.
        valid = y.notna()
        X, y = X[valid], y[valid]
        if len(y) == 0:
            raise ValueError(
                f"The target column '{target_column}' contains no valid values. "
                "Choose a different column."
            )

        detected = detect_problem_type(y)
        resolved = detected if problem_type == "auto" else problem_type

        if resolved == "regression":
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna()
            X, y = X[valid], y[valid]
            if len(y) == 0:
                raise ValueError(
                    f"The target column '{target_column}' cannot be converted to numbers. "
                    "Use a numeric column for regression."
                )
        else:
            if not pd.api.types.is_numeric_dtype(y):
                y = y.astype(str)
            else:
                # mixed-type labels (e.g. ints and strings) break sklearn sorting
                try:
                    sorted(pd.unique(y))
                except TypeError:
                    y = y.astype(str)
            n_classes = int(y.nunique(dropna=True))
            if n_classes < 2:
                raise ValueError(
                    f"The target column '{target_column}' contains only one unique value. "
                    "Choose a column with at least two different values."
                )

        n_samples = len(y)
        random_state = int(config.get("random_state") or RANDOM_STATE)
        test_size = float(config.get("test_size") or 0.2)
        if not 0.05 <= test_size <= 0.9:
            test_size = 0.2

        eval_note: Optional[str] = None
        X_test: Optional[pd.DataFrame] = None
        y_test: Optional[pd.Series] = None

        if n_samples < 4:
            X_train, y_train = X, y
            eval_note = (
                "This dataset is too small for a train/test split, so the results "
                "are measured on the training data."
            )
        else:
            stratify = None
            if resolved == "classification":
                counts = y.value_counts(dropna=True)
                if len(counts) > 1 and counts.min() >= 2:
                    stratify = y
                elif len(counts) > 1 and counts.min() < 2:
                    eval_note = (
                        "At least one class has a single record, so the data was split "
                        "without stratification."
                    )
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=stratify
                )
            except ValueError:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                except ValueError:
                    X_train, y_train = X, y
                    X_test = y_test = None
                    eval_note = (
                        "The dataset is too small to split safely, so the results are "
                        "measured on the training data."
                    )

        # Fit preprocessing on the training split only.
        preprocessor = DataPreprocessor()
        X_train_arr = preprocessor.fit_transform(X_train)
        eval_arr = None
        if X_test is not None and len(X_test) > 0:
            eval_arr = preprocessor.transform(X_test)
        eval_on = "test" if eval_arr is not None else "train"
        if eval_arr is None:
            eval_arr, eval_y = X_train_arr, y_train
        else:
            eval_y = y_test

        feature_names: List[str] = (
            list(preprocessor.feature_names_out)
            if preprocessor.feature_names_out is not None
            else [str(c) for c in X.columns]
        )
        feature_meta = _build_feature_meta(preprocessor, feature_names)

        # ---- estimator -------------------------------------------------
        max_depth = _optional_int(config.get("max_depth"), "max_depth")
        if max_depth is not None and max_depth > MAX_DEPTH_LIMIT:
            raise ValueError(f"'max_depth' cannot be larger than {MAX_DEPTH_LIMIT}.")
        min_samples_leaf = _required_int(
            config.get("min_samples_leaf") if config.get("min_samples_leaf") not in (None, "") else 1,
            "min_samples_leaf",
        )
        min_samples_split = _required_int(
            config.get("min_samples_split") if config.get("min_samples_split") not in (None, "") else 2,
            "min_samples_split",
            minimum=2,
        )
        max_leaf_nodes = _optional_int(config.get("max_leaf_nodes"), "max_leaf_nodes")

        # The requested criterion is validated against the *resolved* task, so a
        # classification criterion (e.g. the UI default ``gini`` with
        # ``problem_type: auto`` on a continuous target) can never reach a
        # regressor, and a regression criterion can never reach a classifier.
        criterion = resolve_criterion(config.get("criterion"), resolved)

        if algorithm == "decision_tree":
            if resolved == "classification":
                estimator: Any = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state,
                )
            else:
                estimator = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state,
                )
        else:
            n_estimators = _required_int(
                config.get("n_estimators") if config.get("n_estimators") not in (None, "") else 10,
                "n_estimators",
            )
            if n_estimators > MAX_TREES:
                raise ValueError(f"'n_estimators' cannot be larger than {MAX_TREES}.")
            max_features = config.get("max_features") or "sqrt"
            if isinstance(max_features, str):
                if max_features in ("all", "None", "none", ""):
                    max_features = None
                elif max_features not in ("sqrt", "log2"):
                    raise ValueError(
                        f"max_features must be one of: {', '.join(MAX_FEATURES_OPTIONS)}."
                    )
            rf_kwargs = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1,
            )
            if resolved == "classification":
                estimator = RandomForestClassifier(criterion=criterion, **rf_kwargs)
            else:
                estimator = RandomForestRegressor(criterion=criterion, **rf_kwargs)

        estimator.fit(X_train_arr, y_train)
        y_pred = estimator.predict(eval_arr)

        # ---- evaluation ------------------------------------------------
        metrics: Dict[str, Any] = {}
        if resolved == "classification":
            metrics = {
                "accuracy": _safe_float(accuracy_score(eval_y, y_pred)),
                "precision": _safe_float(
                    precision_score(eval_y, y_pred, average="weighted", zero_division=0)
                ),
                "recall": _safe_float(
                    recall_score(eval_y, y_pred, average="weighted", zero_division=0)
                ),
                "f1_score": _safe_float(f1_score(eval_y, y_pred, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(eval_y, y_pred).tolist(),
            }
        else:
            mse = mean_squared_error(eval_y, y_pred)
            metrics = {
                "mae": _safe_float(mean_absolute_error(eval_y, y_pred)),
                "mse": _safe_float(mse),
                "rmse": _safe_float(np.sqrt(mse)),
                "r2_score": _safe_float(r2_score(eval_y, y_pred))
                if len(set(np.asarray(eval_y))) > 1
                else None,
            }

        # ---- tree extraction (actual fitted structure) -----------------
        target_names: Optional[List[str]] = None
        if resolved == "classification" and hasattr(estimator, "classes_"):
            target_names = [str(c) for c in estimator.classes_]

        result: Dict[str, Any] = {
            "model_id": str(uuid.uuid4()),
            "type": algorithm,
            "problem_type": resolved,
            "problem_type_detected": detected,
            "target_column": target_column,
            # criterion actually handed to scikit-learn for the resolved task
            "criterion": criterion,
            "metrics": metrics,
            "feature_importances": [float(v) for v in estimator.feature_importances_],
            "feature_names": feature_names,
            "target_names": target_names,
            "feature_columns": [str(c) for c in X.columns],
            "n_train": int(len(y_train)),
            "n_test": int(len(X_test)) if X_test is not None else 0,
            "n_samples": int(n_samples),
            "eval_on": eval_on,
            "eval_note": eval_note,
            "config": config,
            # kept server-side for later prediction / decision path requests
            "pipeline": preprocessor,
            "estimator": estimator,
        }

        if algorithm == "decision_tree":
            result["tree"] = TreeExtractor.extract_tree(
                estimator, feature_names, target_names, feature_meta
            )
        else:
            result["trees"] = [
                TreeExtractor.extract_tree(sub, feature_names, target_names, feature_meta)
                for sub in estimator.estimators_
            ]
            result["n_estimators"] = len(estimator.estimators_)

        return result
