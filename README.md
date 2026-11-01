# Eyeris-Find: Iris Species Classification Engine

Developed a supervised machine learning classification engine to categorize Iris species based on sepal and petal morphometric dimensions. Handled data preprocessing, analyzed linear boundaries, and optimized predictive model accuracy across non-linearly separable feature classes to maximize model generalization.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Historical Data Audit](#dataset--historical-data-audit)
3. [Linear Separability Analysis](#linear-separability-analysis)
4. [Methodology & Architecture](#methodology--architecture)
5. [Model Benchmarking & Comparison](#model-benchmarking--comparison)
6. [Best Model Selection & Generalization Rationale](#best-model-selection--generalization-rationale)
7. [Visual Artifacts](#visual-artifacts)
8. [Installation & Setup](#installation--setup)
9. [CLI Usage Guide](#cli-usage-guide)
10. [Test Suite](#test-suite)
11. [Repository Structure](#repository-structure)

---

## Project Overview

The objective of **Eyeris-Find** is to classify iris specimens into one of three species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) using four morphometric continuous measurements:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

While *Iris setosa* is strictly linearly separable from the other two species, *Iris versicolor* and *Iris virginica* exhibit significant overlap in morphological dimensions. This engine evaluates 9 candidate learning algorithms, quantifies decision boundaries, applies rigorous cross-validation and hyperparameter tuning, and selects the most robust model for production deployment.

---

## Dataset & Historical Data Audit

### UCI `iris.data` vs `bezdekIris.data`
The original UCI repository provides two data files:
1. `iris.data`: Contains transcription errors from R. A. Fisher's 1936 paper *The Use of Multiple Measurements in Taxonomic Problems*.
2. `bezdekIris.data`: The corrected canonical dataset curated by James C. Bezdek (1988) restoring Fisher's original measurements.

### Identified Discrepancies
A line-by-line programmatic audit verifies the exact discrepancies:
- **Row 35 (Index 34)**:
  - `iris.data`: `4.9, 3.1, 1.5, 0.1, Iris-setosa`
  - `bezdekIris.data`: `4.9, 3.1, 1.5, 0.2, Iris-setosa` *(Petal width corrected from 0.1 to 0.2)*
- **Row 38 (Index 37)**:
  - `iris.data`: `4.9, 3.1, 1.5, 0.1, Iris-setosa` *(Erroneous duplicate of row 35)*
  - `bezdekIris.data`: `4.9, 3.6, 1.4, 0.1, Iris-setosa` *(Sepal width corrected to 3.6, petal length to 1.4)*

**Decision**: `bezdekIris.data` is adopted as the canonical reference source for training and testing.

### Data Audit Summary
- **Total Samples**: 150 instances
- **Features**: 4 continuous numerical features (`float64`)
- **Missing Values**: 0 missing values across all columns
- **Class Balance**: Perfectly balanced with exactly 50 instances (33.33%) per species
- **Duplicates**: Exactly 1 pair of biological duplicates (Rows 101 and 142 in *Iris virginica*: `5.8, 2.7, 5.1, 1.9`). Retained as legitimate distinct physical specimens measured at 1 decimal place resolution.

---

## Linear Separability Analysis

We mathematically evaluated pairwise separability using both Perceptron and linear classifiers (Logistic Regression and Linear Support Vector Machines):

| Pairwise Comparison | Perceptron Accuracy | Logistic Regression | Linear SVM | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Setosa vs. Versicolor** | **100.0%** | **100.0%** | **100.0%** | **Linearly Separable** |
| **Setosa vs. Virginica** | **100.0%** | **100.0%** | **100.0%** | **Linearly Separable** |
| **Versicolor vs. Virginica** | 97.0% | 96.0% | 96.0% | **NOT Linearly Separable** |

### Key Conclusions:
1. *Iris setosa* possesses distinctly shorter, narrower petals that form a disjoint convex hull from the other species. It can be classified with 100% accuracy using simple linear hyperplanes.
2. *Iris versicolor* and *Iris virginica* occupy adjacent, slightly overlapping distributions in morphometric feature space (specifically petal length and width). They cannot be separated without soft margin slack or non-linear decision boundaries.

---

## Methodology & Architecture

1. **Stratified Split**: 80/20 train/test split (120 training samples, 30 held-out test samples) stratified by species with fixed seed (`random_state=42`).
2. **Leakage Prevention**: All feature scaling (`StandardScaler`) is encapsulated inside `sklearn.pipeline.Pipeline` objects. The scaler is fit strictly on training folds during cross-validation.
3. **Hyperparameter Tuning**: 5-fold stratified `GridSearchCV` on the training partition optimizes macro F1.
4. **Generalization Validation**: Evaluated on the training set using 10-fold cross-validation repeated 3 times (`RepeatedStratifiedKFold`, 30 fits per model) to quantify mean performance, fold-to-fold variance ($\sigma$), and train-vs-test generalization gap.
5. **Strict Test Isolation**: The held-out test set (30 samples) is touched strictly once after model selection to generate an unbiased estimate of generalization performance.

---

## Model Benchmarking & Comparison

All 9 model pipelines were tuned and benchmarked on the training set (sorted by Cross-Validation Macro F1):

| Model Architecture | Best Hyperparameters | CV Accuracy (Mean ± Std) | CV Macro F1 (Mean ± Std) | Train vs. Val Gap | Held-out Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear SVM** | `{'C': 0.1}` | **0.9694 ± 0.0402** | **0.9690 ± 0.0408** | **0.0046** | **93.33%** |
| **RBF SVM** | `{'C': 1.0, 'gamma': 0.1}` | 0.9611 ± 0.0468 | 0.9606 ± 0.0473 | 0.0167 | 93.33% |
| **Logistic Regression** | `{'C': 10.0, 'solver': 'lbfgs'}` | 0.9611 ± 0.0558 | 0.9605 ± 0.0567 | 0.0170 | 93.33% |
| **Decision Tree** | `{'criterion': 'gini', 'min_samples_split': 5}` | 0.9611 ± 0.0468 | 0.9602 ± 0.0483 | 0.0231 | 93.33% |
| **Random Forest** | `{'n_estimators': 50, 'min_samples_split': 5}` | 0.9583 ± 0.0417 | 0.9577 ± 0.0423 | 0.0281 | 93.33% |
| **KNN** | `{'metric': 'euclidean', 'n_neighbors': 3}` | 0.9528 ± 0.0556 | 0.9523 ± 0.0559 | 0.0102 | 96.67% |
| **Polynomial SVM** | `{'C': 10.0, 'degree': 3, 'gamma': 'scale'}` | 0.9528 ± 0.0634 | 0.9518 ± 0.0648 | 0.0293 | 93.33% |
| **Gradient Boosting** | `{'n_estimators': 50, 'learning_rate': 0.01}` | 0.9500 ± 0.0461 | 0.9489 ± 0.0476 | 0.0448 | 93.33% |
| **MLP (Neural Net)** | `{'hidden_layer_sizes': (50,), 'activation': 'tanh'}` | 0.8472 ± 0.1256 | 0.8459 ± 0.1266 | 0.0022 | 86.67% |

---

## Best Model Selection & Generalization Rationale

### Selection Score Formulation
Rather than optimizing solely for raw empirical training accuracy, our selection criterion explicitly favors generalization:
$$\text{Score} = \text{CV F1}_{\text{macro}} - 0.5 \times \sigma_{\text{CV}} - 0.5 \times |\text{Train-Val Gap}|$$

When models perform within 1% of the top selection score, the principle of **parsimony (Occam's Razor)** is applied to select the least complex architecture.

### Selected Model: **Linear Support Vector Machine (SVC, $C=0.1$)**
- **CV Accuracy**: **96.94% ± 4.02%**
- **CV Macro F1**: **96.90% ± 4.08%**
- **Generalization Gap**: **0.46%** (Train accuracy 97.41% vs. Validation 96.94%)
- **Test Set Accuracy**: **93.33%** (28/30 correct)
- **Test Set Macro F1**: **93.33%**

### Confusion Matrix on Held-Out Test Set (30 samples)
| Actual \ Predicted | Setosa | Versicolor | Virginica | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Iris setosa** | **10** | 0 | 0 | **100.0%** |
| **Iris versicolor** | 0 | **9** | 1 | **90.0%** |
| **Iris virginica** | 0 | 1 | **9** | **90.0%** |
| **Precision** | **100.0%** | **90.0%** | **90.0%** | |

---

## Visual Artifacts

Generated visual reports are saved to `reports/figures/`:

- **Class Balance**: `reports/figures/class_distribution.png`
- **Pairwise Feature Scatter**: `reports/figures/pair_plot.png`
- **Correlation Heatmap**: `reports/figures/correlation_heatmap.png`
- **Morphometric Boxplots**: `reports/figures/feature_boxplots.png`
- **PCA 2D Projection**: `reports/figures/pca_2d_projection.png` (PC1: 73.0%, PC2: 22.9% explained variance)
- **Pairwise Boundaries**:
  - Setosa vs. Versicolor: `reports/figures/boundary_setosa_vs_versicolor.png`
  - Setosa vs. Virginica: `reports/figures/boundary_setosa_vs_virginica.png`
  - Versicolor vs. Virginica: `reports/figures/boundary_versicolor_vs_virginica.png`
- **Model Benchmark Comparison**: `reports/figures/model_comparison.png`
- **Test Confusion Matrix**: `reports/figures/confusion_matrix_best.png`

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.14.3)
- Git

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/The-Pehh-shhh/Eyeris-find.git
cd Eyeris-find

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

---

## CLI Usage Guide

The repository includes a ready-to-use prediction CLI powered by the trained pipeline:

```bash
# Predict Setosa
python -m src.predict --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
# Output: Predicted species: Iris-setosa

# Predict Versicolor
python -m src.predict --sl 6.0 --sw 2.9 --pl 4.5 --pw 1.5
# Output: Predicted species: Iris-versicolor

# Predict Virginica
python -m src.predict --sl 6.5 --sw 3.0 --pl 5.5 --pw 2.0
# Output: Predicted species: Iris-virginica

# Alternatively, invoke through __main__.py
python -m src --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
```

### Parameter Reference
| Argument | Type | Description |
| :--- | :--- | :--- |
| `--sl` | float | Sepal length in centimeters |
| `--sw` | float | Sepal width in centimeters |
| `--pl` | float | Petal length in centimeters |
| `--pw` | float | Petal width in centimeters |
| `--model` | str (optional) | Custom path to saved model (default: `models/best_model.joblib`) |

---

## Re-training the Models

To execute the entire training, audit, EDA, boundary analysis, hyperparameter tuning, and model comparison workflow from scratch:

```bash
python -m src.train
```

Outputs will be saved directly into `models/` (`best_model.joblib`, `metrics.json`) and `reports/figures/`.

---

## Test Suite

Run the full automated test suite covering data loading, pipeline integrity, accuracy thresholds, and prediction CLI:

```bash
pytest
```

All 17 tests will run and pass:
- `TestDataLoading`: Shape, column names, null checks, class balance, dtypes, feature ranges.
- `TestPipeline`: Stratified train/test splitting, leakage-free pipeline fitting, and multi-model configuration validation.
- `TestAccuracyThreshold`: Asserts minimum test accuracy $\ge 93\%$ and verifies 100% accuracy on *Iris setosa*.
- `TestPredict`: Evaluates CLI inference consistency and valid class domain mapping.
- `TestAudit`: Validates programmatic audit output schemas.

---

## Repository Structure

```
eyerisfind/
│
├── iris/                               # Raw dataset directory
│   ├── bezdekIris.data                 # Canonical corrected dataset (Bezdek 1988)
│   ├── iris.data                       # Legacy UCI dataset with known transcription errors
│   ├── iris.names                      # Dataset documentation and summary statistics
│   └── Index                           # UCI index file
│
├── models/                             # Trained model artifacts & metrics
│   ├── best_model.joblib               # Optimized Linear SVM pipeline
│   └── metrics.json                    # Benchmark comparison and cross-validation logs
│
├── notebooks/                          # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb   # Interactive analysis & inspection walkthrough
│
├── reports/
│   └── figures/                        # Generated visual artifacts
│       ├── boundary_setosa_vs_versicolor.png
│       ├── boundary_setosa_vs_virginica.png
│       ├── boundary_versicolor_vs_virginica.png
│       ├── class_distribution.png
│       ├── confusion_matrix_best.png
│       ├── correlation_heatmap.png
│       ├── feature_boxplots.png
│       ├── model_comparison.png
│       ├── pair_plot.png
│       └── pca_2d_projection.png
│
├── src/                                # Core production engine modules
│   ├── __init__.py                     # Package initialization
│   ├── __main__.py                     # Direct CLI dispatch
│   ├── data_loader.py                  # Dataset loading, auditing, and splitting
│   ├── evaluation.py                   # Repeated CV, GridSearchCV, and metrics logging
│   ├── models.py                       # 9 Pipeline model definitions & parameter grids
│   ├── plotting.py                     # Headless publication-quality visualization routines
│   ├── predict.py                      # Standalone CLI prediction interface
│   ├── preprocessing.py                # Preprocessing transformer pipelines
│   └── train.py                        # Orchestrator running the complete ML pipeline
│
├── tests/                              # Pytest test suite
│   ├── __init__.py
│   └── test_iris.py                    # 17 Unit, integration, and threshold test cases
│
├── .gitignore                          # Ignored caches, checkpoints, and venv
├── pytest.ini                          # Pytest root configuration
├── README.md                           # Comprehensive documentation & results
└── requirements.txt                    # Pinned package dependencies
```
