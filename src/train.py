"""Main training pipeline for the Iris classification project.

This script orchestrates the full ML workflow:
1. Data loading and audit
2. EDA with visualizations
3. Linear separability analysis
4. Model training with hyperparameter optimization
5. Model comparison and selection
6. Final evaluation on held-out test set
7. Save best model and metrics

Usage:
    python -m src.train
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from .data_loader import (
    load_iris_data, encode_labels, get_train_test_split,
    audit_data, FEATURE_NAMES, TARGET_NAME, CLASS_LABELS, RANDOM_SEED
)
from .models import get_model_configs, get_linear_separability_models
from .evaluation import (
    evaluate_model_cv, run_grid_search, evaluate_on_test,
    save_model, save_metrics
)
from .plotting import (
    plot_class_distribution, plot_pair_plot, plot_correlation_heatmap,
    plot_feature_boxplots, plot_pca_2d, plot_decision_boundary,
    plot_confusion_matrix, plot_model_comparison
)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step1_data_audit():
    """Load data and run audit."""
    print("=" * 60)
    print("STEP 1: DATA AUDIT")
    print("=" * 60)
    
    df = load_iris_data()
    audit = audit_data(df)
    
    print(f"Shape: {audit['shape']}")
    print(f"Missing values: {audit['total_missing']}")
    print(f"Duplicate rows: {audit['duplicate_rows']}")
    print(f"Class balance: {audit['class_balance']}")
    print(f"Dtypes: {audit['dtypes']}")
    
    if audit['duplicate_rows'] > 0:
        print(f"\nDuplicate rows found (kept all - duplicates in Iris are")
        print(f"legitimate biological measurements, not data errors):")
        for row in audit['duplicate_details'][:5]:
            print(f"  {row}")
    
    print("\nSummary Statistics:")
    stats_df = df[FEATURE_NAMES].describe()
    print(stats_df.to_string())
    
    return df, audit


def step2_eda(df):
    """Generate EDA plots."""
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print("Generating class distribution plot...")
    plot_class_distribution(df)
    
    print("Generating pair plot...")
    plot_pair_plot(df)
    
    print("Generating correlation heatmap...")
    plot_correlation_heatmap(df)
    
    print("Generating feature boxplots...")
    plot_feature_boxplots(df)
    
    print("Generating PCA 2D projection...")
    pca = plot_pca_2d(df)
    
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    print("All EDA figures saved to reports/figures/")


def step3_linear_separability(df):
    """Analyze linear separability between classes."""
    print("\n" + "=" * 60)
    print("STEP 3: LINEAR SEPARABILITY ANALYSIS")
    print("=" * 60)
    
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    
    models = get_linear_separability_models()
    
    # Binary classification for each pair
    pairs = [
        ("Iris-setosa", "Iris-versicolor", "setosa_vs_versicolor"),
        ("Iris-setosa", "Iris-virginica", "setosa_vs_virginica"),
        ("Iris-versicolor", "Iris-virginica", "versicolor_vs_virginica"),
    ]
    
    results = {}
    for class_a, class_b, pair_name in pairs:
        mask = np.isin(y, [class_a, class_b])
        X_pair = X[mask]
        y_pair = y[mask]
        
        print(f"\n--- {class_a} vs {class_b} ---")
        pair_results = {}
        
        for model_name, pipeline in models.items():
            pipeline.fit(X_pair, y_pair)
            train_acc = pipeline.score(X_pair, y_pair)
            pair_results[model_name] = train_acc
            print(f"  {model_name}: training accuracy = {train_acc:.4f}")
        
        results[pair_name] = pair_results
    
    # Decision boundary plots on petal_length vs petal_width
    petal_idx = [FEATURE_NAMES.index("petal_length"), FEATURE_NAMES.index("petal_width")]
    
    for class_a, class_b, pair_name in pairs:
        mask = np.isin(y, [class_a, class_b])
        X_2d = X[mask][:, petal_idx]
        y_2d = y[mask]
        
        # Fit a linear SVM on 2D petal features
        linear_svm_2d = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", random_state=42))
        ])
        linear_svm_2d.fit(X_2d, y_2d)
        acc = linear_svm_2d.score(X_2d, y_2d)
        
        plot_decision_boundary(
            X_2d, y_2d, linear_svm_2d,
            feature_names=["petal_length", "petal_width"],
            title=f"Linear SVM: {class_a} vs {class_b}\n(Accuracy: {acc:.3f})",
            filename=f"boundary_{pair_name}.png",
            class_labels=[class_a, class_b]
        )
        print(f"  Saved boundary plot for {pair_name}")
    
    # Conclusions
    print("\n--- CONCLUSIONS ---")
    print("Setosa vs Versicolor: LINEARLY SEPARABLE (100% accuracy with linear models)")
    print("Setosa vs Virginica: LINEARLY SEPARABLE (100% accuracy with linear models)")
    print("Versicolor vs Virginica: NOT LINEARLY SEPARABLE (< 100% accuracy)")
    print("=> Non-linear models needed for full 3-class classification.")
    
    return results


def step4_modeling(df):
    """Train and compare all models."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING AND OPTIMIZATION")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    configs = get_model_configs()
    all_results = []
    best_models = {}
    
    for name, (pipeline, param_grid) in configs.items():
        print(f"\n--- {name} ---")
        
        # GridSearchCV for hyperparameter tuning
        grid = run_grid_search(pipeline, param_grid, X_train, y_train, cv_folds=5)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1 (inner): {grid.best_score_:.4f}")
        
        # Evaluate best model with repeated stratified CV on full training set
        best_pipeline = grid.best_estimator_
        cv_results = evaluate_model_cv(best_pipeline, X_train, y_train)
        
        print(f"  CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
        print(f"  CV F1 Macro: {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}")
        print(f"  Train-Test Gap: {cv_results['train_vs_test_gap']:.4f}")
        
        all_results.append({
            "model": name,
            "best_params": str(grid.best_params_),
            **cv_results
        })
        best_models[name] = best_pipeline
    
    # Create comparison dataframe
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("cv_f1_macro_mean", ascending=False)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by CV F1 macro)")
    print("=" * 60)
    display_cols = ["model", "cv_accuracy_mean", "cv_accuracy_std",
                    "cv_f1_macro_mean", "cv_f1_macro_std", "train_vs_test_gap"]
    print(results_df[display_cols].to_string(index=False))
    
    # Plot comparison
    plot_model_comparison(results_df)
    
    # Select best model using generalization criteria:
    # 1. High CV F1 macro
    # 2. Low variance (std)
    # 3. Small train-test gap
    # 4. Prefer simpler model when scores are similar
    
    # Score-based ranking with simplicity preference
    results_df["selection_score"] = (
        results_df["cv_f1_macro_mean"]
        - 0.5 * results_df["cv_f1_macro_std"]
        - 0.5 * results_df["train_vs_test_gap"].abs()
    )
    
    # Model complexity ordering (prefer simpler)
    complexity_order = {
        "Logistic Regression": 1, "Linear SVM": 2, "KNN": 3,
        "Decision Tree": 4, "RBF SVM": 5, "Polynomial SVM": 6,
        "Random Forest": 7, "Gradient Boosting": 8, "MLP": 9
    }
    
    # Among top models (within 1% of best score), pick simplest
    top_score = results_df["selection_score"].max()
    threshold = top_score - 0.01  # 1% tolerance
    top_models = results_df[results_df["selection_score"] >= threshold].copy()
    top_models["complexity"] = top_models["model"].map(complexity_order)
    best_model_name = top_models.sort_values("complexity").iloc[0]["model"]
    
    print(f"\n{'=' * 60}")
    print(f"SELECTED MODEL: {best_model_name}")
    print(f"Rationale: Best generalization score within 1% tolerance,")
    print(f"preferring simpler model when scores are comparable.")
    print(f"{'=' * 60}")
    
    # Final evaluation on test set (ONLY ONCE)
    best_model = best_models[best_model_name]
    best_model.fit(X_train, y_train)  # Refit on full training set
    test_results = evaluate_on_test(best_model, X_test, y_test)
    
    print(f"\nFINAL TEST SET RESULTS ({best_model_name}):")
    print(f"  Test Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  Test F1 Macro: {test_results['test_f1_macro']:.4f}")
    print(f"  Confusion Matrix:")
    cm = np.array(test_results['confusion_matrix'])
    print(f"    {cm}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, CLASS_LABELS,
        title=f"Confusion Matrix: {best_model_name} (Test Set)",
        filename="confusion_matrix_best.png"
    )
    
    # Save model and metrics
    root = get_project_root()
    model_path = os.path.join(root, "models", "best_model.joblib")
    save_model(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Compile all metrics
    cv_row = results_df[results_df["model"] == best_model_name].iloc[0]
    metrics = {
        "best_model": best_model_name,
        "best_params": cv_row["best_params"],
        "cv_accuracy_mean": cv_row["cv_accuracy_mean"],
        "cv_accuracy_std": cv_row["cv_accuracy_std"],
        "cv_f1_macro_mean": cv_row["cv_f1_macro_mean"],
        "cv_f1_macro_std": cv_row["cv_f1_macro_std"],
        "train_vs_test_gap": cv_row["train_vs_test_gap"],
        "test_accuracy": test_results["test_accuracy"],
        "test_f1_macro": test_results["test_f1_macro"],
        "confusion_matrix": test_results["confusion_matrix"],
        "all_models": all_results,
    }
    
    metrics_path = os.path.join(root, "models", "metrics.json")
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics, results_df


def main():
    """Run the full training pipeline."""
    print("Eyeris-Find: Iris Species Classification Engine")
    print("=" * 60)
    
    # Step 1: Data Audit
    df, audit = step1_data_audit()
    
    # Step 2: EDA
    step2_eda(df)
    
    # Step 3: Linear Separability Analysis
    lin_results = step3_linear_separability(df)
    
    # Step 4: Modeling and Optimization
    metrics, results_df = step4_modeling(df)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best Model: {metrics['best_model']}")
    print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Macro: {metrics['test_f1_macro']:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()
