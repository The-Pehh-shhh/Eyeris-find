"""Data loading and preprocessing utilities for the Iris dataset."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Column names as specified in iris.names
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAME = "species"
COLUMN_NAMES = FEATURE_NAMES + [TARGET_NAME]

# Class labels
CLASS_LABELS = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Random seed for reproducibility
RANDOM_SEED = 42


def get_data_path():
    """Return the path to the canonical dataset (bezdekIris.data).
    
    We use bezdekIris.data because it contains corrections to two known
    errors in the original UCI iris.data file:
      - Row 35: petal_width corrected from 0.1 to 0.2
      - Row 38: sepal_width corrected from 3.1 to 3.6,
                petal_length corrected from 1.5 to 1.4
    These corrections align with Fisher's original 1936 paper.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "iris", "bezdekIris.data")


def load_iris_data(filepath=None):
    """Load the Iris dataset from CSV.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the data file. Defaults to bezdekIris.data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sepal_length, sepal_width, petal_length,
        petal_width, species.
    """
    if filepath is None:
        filepath = get_data_path()
    
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    
    # Remove any trailing blank rows (the file has trailing newlines)
    df = df.dropna(how="all").reset_index(drop=True)
    
    # Validate dtypes
    for col in FEATURE_NAMES:
        df[col] = df[col].astype(float)
    
    # Validate class labels
    assert set(df[TARGET_NAME].unique()) == set(CLASS_LABELS), \
        f"Unexpected labels: {df[TARGET_NAME].unique()}"
    
    # Validate shape
    assert df.shape == (150, 5), f"Unexpected shape: {df.shape}"
    
    return df


def encode_labels(df):
    """Encode species labels as integers.
    
    Returns
    -------
    tuple
        (df_encoded, label_encoder) where species column is integer-encoded.
    """
    le = LabelEncoder()
    le.fit(CLASS_LABELS)  # Fit with sorted order for consistency
    df = df.copy()
    df["species_encoded"] = le.transform(df[TARGET_NAME])
    return df, le


def get_train_test_split(df, test_size=0.2, random_state=RANDOM_SEED):
    """Stratified train/test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with feature columns and 'species' column.
    test_size : float
        Fraction of data for test set.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAME].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def audit_data(df):
    """Run data quality checks and return audit report dict.
    
    Returns
    -------
    dict
        Audit results including missing values, duplicates, class balance, etc.
    """
    report = {}
    
    # Missing values
    report["missing_values"] = df.isnull().sum().to_dict()
    report["total_missing"] = int(df.isnull().sum().sum())
    
    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_details"] = df[df.duplicated(keep=False)].to_dict("records") \
        if report["duplicate_rows"] > 0 else []
    
    # Class balance
    report["class_counts"] = df[TARGET_NAME].value_counts().to_dict()
    report["class_balance"] = {
        k: f"{v / len(df) * 100:.1f}%"
        for k, v in df[TARGET_NAME].value_counts().items()
    }
    
    # Dtypes
    report["dtypes"] = df.dtypes.astype(str).to_dict()
    
    # Basic stats
    report["shape"] = df.shape
    report["summary_stats"] = df[FEATURE_NAMES].describe().to_dict()
    
    return report
