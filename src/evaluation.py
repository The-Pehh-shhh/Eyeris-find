"""Evaluation utilities for model comparison and metric computation."""

import json
import os
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)
import joblib

from .data_loader import RANDOM_SEED


def evaluate_model_cv(pipeline, X, y, cv_folds=10, n_repeats=3):
    """Evaluate a pipeline using repeated stratified k-fold CV.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Complete pipeline including preprocessing and classifier.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of folds for stratified k-fold.
    n_repeats : int
        Number of repetitions for repeated CV.
    
    Returns
    -------
    dict
        Cross-validation results with mean/std for accuracy and F1.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED
    )
    
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }
    
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    return {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "train_accuracy_mean": float(np.mean(cv_results["train_accuracy"])),
        "train_accuracy_std": float(np.std(cv_results["train_accuracy"])),
        "train_f1_macro_mean": float(np.mean(cv_results["train_f1_macro"])),
        "train_vs_test_gap": float(
            np.mean(cv_results["train_accuracy"]) - np.mean(cv_results["test_accuracy"])
        ),
    }


def run_grid_search(pipeline, param_grid, X_train, y_train, cv_folds=5):
    """Run GridSearchCV for hyperparameter optimization.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Model pipeline.
    param_grid : dict
        Hyperparameter grid.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    cv_folds : int
        Number of CV folds for inner loop.
    
    Returns
    -------
    GridSearchCV
        Fitted grid search object.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        refit=True,
        n_jobs=-1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_on_test(model, X_test, y_test):
    """Final evaluation on the held-out test set.
    
    Parameters
    ----------
    model : fitted estimator
        Trained model or pipeline.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True test labels.
    
    Returns
    -------
    dict
        Test metrics including accuracy, F1, confusion matrix, and report.
    """
    y_pred = model.predict(X_test)
    
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_model(model, filepath):
    """Save a trained model using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load a saved model."""
    return joblib.load(filepath)


def save_metrics(metrics, filepath):
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
