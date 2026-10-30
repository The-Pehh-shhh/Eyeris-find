"""Command-line interface for Iris species prediction.

Usage:
    python -m src.predict --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
"""

import argparse
import os
import sys
import numpy as np

from .evaluation import load_model
from .data_loader import FEATURE_NAMES


def get_default_model_path():
    """Return default path to the saved best model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "best_model.joblib")


def predict(sepal_length, sepal_width, petal_length, petal_width,
            model_path=None):
    """Predict iris species from measurements.
    
    Parameters
    ----------
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
    model_path : str, optional
        Path to saved model. Defaults to models/best_model.joblib.
    
    Returns
    -------
    str
        Predicted species name.
    """
    if model_path is None:
        model_path = get_default_model_path()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the training pipeline first."
        )
    
    model = load_model(model_path)
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(X)
    return prediction[0]


def main():
    parser = argparse.ArgumentParser(
        description="Predict Iris species from sepal/petal measurements."
    )
    parser.add_argument("--sl", type=float, required=True,
                        help="Sepal length in cm")
    parser.add_argument("--sw", type=float, required=True,
                        help="Sepal width in cm")
    parser.add_argument("--pl", type=float, required=True,
                        help="Petal length in cm")
    parser.add_argument("--pw", type=float, required=True,
                        help="Petal width in cm")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to saved model (default: models/best_model.joblib)")
    
    args = parser.parse_args()
    
    try:
        species = predict(args.sl, args.sw, args.pl, args.pw, args.model)
        print(f"Predicted species: {species}")
        return species
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
