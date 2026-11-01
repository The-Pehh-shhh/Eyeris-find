"""Model definitions and training pipelines for Iris classification.

Each function returns an sklearn Pipeline that includes preprocessing
(StandardScaler) followed by the classifier, ensuring no data leakage.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def get_model_configs():
    """Return dict of model name -> (pipeline, param_grid) for GridSearchCV.
    
    All param grids use the 'clf__' prefix because the classifier step
    in each pipeline is named 'clf'.
    
    Returns
    -------
    dict
        {model_name: (Pipeline, param_grid_dict)}
    """
    configs = {}
    
    # --- Baselines ---
    
    configs["Logistic Regression"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs"],
        }
    )
    
    configs["Linear SVM"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", random_state=42))
        ]),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }
    )
    
    # --- Non-linear models ---
    
    configs["RBF SVM"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42))
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
        }
    )
    
    configs["Polynomial SVM"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="poly", random_state=42))
        ]),
        {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__degree": [2, 3, 4],
            "clf__gamma": ["scale", "auto"],
        }
    )
    
    configs["KNN"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        }
    )
    
    configs["Decision Tree"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        {
            "clf__max_depth": [None, 3, 5, 7, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        }
    )
    
    configs["Random Forest"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        }
    )
    
    configs["Gradient Boosting"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        }
    )
    
    configs["MLP"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                random_state=42,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15
            ))
        ]),
        {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 30), (100, 50)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__activation": ["relu", "tanh"],
        }
    )
    
    return configs


def get_linear_separability_models():
    """Return models for linear separability analysis.
    
    Returns
    -------
    dict
        {name: Pipeline} for Perceptron and Logistic Regression.
    """
    models = {
        "Perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Perceptron(max_iter=1000, random_state=42, tol=1e-3))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", random_state=42))
        ]),
    }
    return models
