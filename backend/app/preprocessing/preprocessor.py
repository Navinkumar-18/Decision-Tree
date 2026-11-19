"""Preprocessing pipeline shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..services.datasets import MAX_ONE_HOT_FEATURES


@dataclass
class FeatureMeta:
    """Describes one transformed column produced by the preprocessor."""

    name: str
    column: str
    kind: str
    category: str | None = None


@dataclass
class PreparedFeatures:
    transformer: ColumnTransformer
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    metas: list[FeatureMeta]
    numeric_means: list[float]
    numeric_scales: list[float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = _prepare_frame(frame, self.feature_columns, self.categorical_columns)
        return self.transformer.transform(prepared)

    def original_threshold(self, feature_index: int, threshold: float) -> float:
        """Convert a scaled threshold back to the caller's units."""
        if feature_index >= len(self.numeric_columns):
            return float(threshold)
        scale = self.numeric_scales[feature_index]
        mean = self.numeric_means[feature_index]
        if not np.isfinite(scale) or scale == 0:
            return float(threshold)
        return float(threshold * scale + mean)


def _is_numeric_column(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def _as_category(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _prepare_frame(
    frame: pd.DataFrame, feature_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    prepared = frame.reindex(columns=feature_columns).copy()
    for column in categorical_columns:
        prepared[column] = prepared[column].map(_as_category)
    return prepared


def build_preprocessor(
    frame: pd.DataFrame, feature_columns: list[str]
) -> PreparedFeatures:
    numeric_columns = [column for column in feature_columns if _is_numeric_column(frame[column])]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    if not numeric_columns and not categorical_columns:
        raise ValueError("The dataset has no usable feature columns.")

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        total_categories = 0
        for column in categorical_columns:
            total_categories += int(frame[column].nunique(dropna=True))
        if total_categories > MAX_ONE_HOT_FEATURES:
            raise ValueError(
                "The categorical columns have too many distinct values to encode. "
                "Reduce the number of categories or one-hot encode them before uploading."
            )
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    prepared = _prepare_frame(frame, feature_columns, categorical_columns)
    transformer.fit(prepared)

    metas: list[FeatureMeta] = []
    numeric_means: list[float] = []
    numeric_scales: list[float] = []

    if numeric_columns:
        scaler = transformer.named_transformers_["num"].named_steps["scaler"]
        numeric_means = [float(value) for value in scaler.mean_]
        numeric_scales = [float(value) for value in scaler.scale_]
        for column in numeric_columns:
            metas.append(FeatureMeta(name=column, column=column, kind="numeric"))

    if categorical_columns:
        encoder = transformer.named_transformers_["cat"].named_steps["onehot"]
        for column, categories in zip(categorical_columns, encoder.categories_):
            for category in categories:
                label = str(category)
                metas.append(
                    FeatureMeta(
                        name=f"{column}={label}",
                        column=column,
                        kind="categorical",
                        category=label,
                    )
                )

    return PreparedFeatures(
        transformer=transformer,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        metas=metas,
        numeric_means=numeric_means,
        numeric_scales=numeric_scales,
    )
