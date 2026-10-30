"""Visualization and plotting utilities for the Iris project.

All plots are saved to reports/figures/ by default. Matplotlib backend
is set to 'Agg' for headless environments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_NAMES, TARGET_NAME, CLASS_LABELS

# Plot style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
FIGURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports", "figures"
)


def _ensure_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def plot_class_distribution(df, filename="class_distribution.png"):
    """Bar plot of class counts."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[TARGET_NAME].value_counts()
    colors = sns.color_palette("deep", n_colors=3)
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Iris Species Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    for i, (label, val) in enumerate(counts.items()):
        ax.text(i, val + 0.5, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pair_plot(df, filename="pair_plot.png"):
    """Seaborn pair plot colored by species."""
    _ensure_dir()
    g = sns.pairplot(
        df, hue=TARGET_NAME,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "k", "linewidth": 0.3},
        palette="deep"
    )
    g.figure.suptitle("Iris Feature Pair Plot", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def plot_correlation_heatmap(df, filename="correlation_heatmap.png"):
    """Correlation heatmap for numeric features."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        mask=mask, center=0, square=True, ax=ax,
        linewidths=0.5, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_boxplots(df, filename="feature_boxplots.png"):
    """Per-feature boxplots grouped by species."""
    _ensure_dir()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (col, ax) in enumerate(zip(FEATURE_NAMES, axes.flat)):
        sns.boxplot(data=df, x=TARGET_NAME, y=col, ax=ax, palette="deep",
                    hue=TARGET_NAME, legend=False)
        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_2d(df, filename="pca_2d_projection.png"):
    """PCA 2D projection of the dataset colored by species."""
    _ensure_dir()
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("deep", n_colors=3)
    for i, label in enumerate(CLASS_LABELS):
        mask = y == label
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[colors[i]], label=label, alpha=0.7, s=60,
            edgecolors="k", linewidth=0.3
        )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA 2D Projection of Iris Dataset", fontsize=14, fontweight="bold")
    ax.legend(title="Species", loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return pca


def plot_decision_boundary(X, y, model, feature_names, title, filename,
                           class_labels=None):
    """Plot 2D decision boundary for a fitted model.
    
    Parameters
    ----------
    X : np.ndarray
        2D feature array (already selected features).
    y : np.ndarray
        Labels (string or integer).
    model : fitted estimator
        Must have .predict() method.
    feature_names : list
        Names for x and y axes.
    title : str
        Plot title.
    filename : str
        Output filename.
    class_labels : list, optional
        Class label strings for legend.
    """
    _ensure_dir()
    
    if class_labels is None:
        class_labels = sorted(np.unique(y))
    
    h = 0.02  # mesh step
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Convert string labels to numeric for contourf
    unique_labels = sorted(np.unique(np.concatenate([y, Z])))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    Z_num = np.array([label_map[z] for z in Z]).reshape(xx.shape)
    y_num = np.array([label_map[yi] for yi in y])
    
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_bg = plt.cm.RdYlBu
    cmap_pts = sns.color_palette("deep", n_colors=len(unique_labels))
    
    ax.contourf(xx, yy, Z_num, alpha=0.3, cmap=cmap_bg)
    ax.contour(xx, yy, Z_num, colors="k", linewidths=0.5, alpha=0.5)
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[cmap_pts[i]], label=label, alpha=0.7, s=60,
            edgecolors="k", linewidth=0.3
        )
    
    ax.set_xlabel(feature_names[0].replace("_", " ").title())
    ax.set_ylabel(feature_names[1].replace("_", " ").title())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Species", loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, title, filename):
    """Plot a confusion matrix heatmap."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        square=True, ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_df, filename="model_comparison.png"):
    """Bar plot comparing models on CV accuracy with error bars."""
    _ensure_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    ax = axes[0]
    colors = sns.color_palette("deep", n_colors=len(results_df))
    bars = ax.barh(
        results_df["model"], results_df["cv_accuracy_mean"],
        xerr=results_df["cv_accuracy_std"],
        color=colors, edgecolor="black", linewidth=0.5,
        capsize=3
    )
    ax.set_xlabel("CV Accuracy (mean ± std)")
    ax.set_title("Model Comparison: Accuracy", fontweight="bold")
    ax.set_xlim(0.85, 1.01)
    
    # F1
    ax = axes[1]
    ax.barh(
        results_df["model"], results_df["cv_f1_macro_mean"],
        xerr=results_df["cv_f1_macro_std"],
        color=colors, edgecolor="black", linewidth=0.5,
        capsize=3
    )
    ax.set_xlabel("CV Macro F1 (mean ± std)")
    ax.set_title("Model Comparison: Macro F1", fontweight="bold")
    ax.set_xlim(0.85, 1.01)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
