"""Preprocessing pipelines for the Iris classification project.

All scaling and transformations are wrapped in sklearn Pipelines
to prevent data leakage between train and test sets.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


def create_preprocessing_pipeline():
    """Create a feature preprocessing pipeline.
    
    Uses StandardScaler for all numeric features. The scaler is fitted
    only on training data inside the pipeline, preventing data leakage.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline with StandardScaler.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    return pipeline
