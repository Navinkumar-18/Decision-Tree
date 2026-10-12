from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.column_transformer = None
        self.feature_names_out = None
        self.numeric_features = []
        self.categorical_features = []
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        self.column_transformer = ColumnTransformer(transformers=transformers)
        result = self.column_transformer.fit_transform(X)
        try:
            self.feature_names_out = self.column_transformer.get_feature_names_out()
        except Exception:
            self.feature_names_out = None
        return result
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            return X.values
        return self.column_transformer.transform(X)
