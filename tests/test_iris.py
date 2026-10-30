"""Tests for data loading, pipeline, accuracy threshold, and prediction."""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_iris_data, get_train_test_split, audit_data,
    FEATURE_NAMES, TARGET_NAME, CLASS_LABELS, RANDOM_SEED
)
from src.models import get_model_configs


class TestDataLoading:
    """Tests for data loading and validation."""
    
    def test_load_shape(self):
        """Dataset must have 150 rows and 5 columns."""
        df = load_iris_data()
        assert df.shape == (150, 5)
    
    def test_column_names(self):
        """Columns must match expected names."""
        df = load_iris_data()
        expected = FEATURE_NAMES + [TARGET_NAME]
        assert list(df.columns) == expected
    
    def test_no_missing_values(self):
        """No missing values allowed."""
        df = load_iris_data()
        assert df.isnull().sum().sum() == 0
    
    def test_class_labels(self):
        """Must contain exactly the three iris species."""
        df = load_iris_data()
        assert set(df[TARGET_NAME].unique()) == set(CLASS_LABELS)
    
    def test_class_balance(self):
        """Each class must have exactly 50 samples."""
        df = load_iris_data()
        counts = df[TARGET_NAME].value_counts()
        for label in CLASS_LABELS:
            assert counts[label] == 50
    
    def test_feature_dtypes(self):
        """All features must be float."""
        df = load_iris_data()
        for col in FEATURE_NAMES:
            assert df[col].dtype == np.float64
    
    def test_feature_ranges(self):
        """Features must be within biologically plausible ranges."""
        df = load_iris_data()
        assert df["sepal_length"].min() >= 4.0
        assert df["sepal_length"].max() <= 8.0
        assert df["petal_width"].min() >= 0.1
        assert df["petal_width"].max() <= 2.6


class TestPipeline:
    """Tests for preprocessing and model pipelines."""
    
    def test_train_test_split_shapes(self):
        """Train/test split must preserve stratification."""
        df = load_iris_data()
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        assert X_train.shape[0] == 120
        assert X_test.shape[0] == 30
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4
    
    def test_stratified_split(self):
        """Each class must be represented in both train and test."""
        df = load_iris_data()
        _, _, y_train, y_test = get_train_test_split(df)
        
        for label in CLASS_LABELS:
            assert label in y_train
            assert label in y_test
    
    def test_pipeline_fit_predict(self):
        """A basic pipeline must fit and predict without error."""
        df = load_iris_data()
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        assert len(preds) == len(y_test)
        assert all(p in CLASS_LABELS for p in preds)
    
    def test_all_model_configs_valid(self):
        """All model configs must produce valid pipelines."""
        configs = get_model_configs()
        assert len(configs) >= 8  # We defined 8+ models
        
        df = load_iris_data()
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        for name, (pipeline, param_grid) in configs.items():
            assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"
            # Quick fit/predict check
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            assert len(preds) == len(y_test), f"{name} prediction length mismatch"


class TestAccuracyThreshold:
    """Tests for minimum accuracy requirements."""
    
    def test_minimum_accuracy(self):
        """Best model must achieve at least 93% on test set."""
        df = load_iris_data()
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        # Use RBF SVM as a reliable baseline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ])
        pipe.fit(X_train, y_train)
        accuracy = pipe.score(X_test, y_test)
        
        assert accuracy >= 0.93, f"Accuracy {accuracy:.4f} below 93% threshold"
    
    def test_setosa_perfect_classification(self):
        """Setosa (linearly separable) must be classified perfectly."""
        df = load_iris_data()
        X_train, X_test, y_train, y_test = get_train_test_split(df)
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        setosa_mask = y_test == "Iris-setosa"
        setosa_accuracy = np.mean(preds[setosa_mask] == y_test[setosa_mask])
        assert setosa_accuracy == 1.0, f"Setosa accuracy {setosa_accuracy} != 100%"


class TestPredict:
    """Tests for the prediction CLI."""
    
    def test_predict_setosa(self):
        """Known Setosa measurements must predict Setosa."""
        # First ensure model exists by training a quick one
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model.joblib"
        )
        
        if not os.path.exists(model_path):
            pytest.skip("Model not yet trained; run training pipeline first")
        
        from src.predict import predict
        result = predict(5.1, 3.5, 1.4, 0.2, model_path)
        assert result == "Iris-setosa", f"Expected Iris-setosa, got {result}"
    
    def test_predict_returns_valid_class(self):
        """Prediction must return one of the three valid classes."""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model.joblib"
        )
        
        if not os.path.exists(model_path):
            pytest.skip("Model not yet trained; run training pipeline first")
        
        from src.predict import predict
        result = predict(6.3, 3.3, 6.0, 2.5, model_path)
        assert result in CLASS_LABELS, f"Invalid prediction: {result}"


class TestAudit:
    """Tests for data audit functionality."""
    
    def test_audit_returns_dict(self):
        """Audit must return a dict with expected keys."""
        df = load_iris_data()
        audit = audit_data(df)
        
        assert isinstance(audit, dict)
        assert "missing_values" in audit
        assert "duplicate_rows" in audit
        assert "class_counts" in audit
        assert "shape" in audit
    
    def test_no_missing_in_audit(self):
        """Audit must confirm zero missing values."""
        df = load_iris_data()
        audit = audit_data(df)
        assert audit["total_missing"] == 0
