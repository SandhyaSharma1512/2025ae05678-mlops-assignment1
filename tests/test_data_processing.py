import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest
from heart_disease.data import download_dataset, prepare_dataset, RAW_PATH, PROCESSED_PATH, normalize_columns

def test_download_dataset():
    """
    Test if the dataset is downloaded successfully.
    """
    download_dataset(RAW_PATH)
    assert RAW_PATH.exists(), "Raw dataset file was not downloaded."

def test_prepare_dataset():
    """
    Test if the dataset is prepared successfully.
    """
    prepare_dataset(RAW_PATH, PROCESSED_PATH)
    assert PROCESSED_PATH.exists(), "Processed dataset file was not created."

def test_normalize_columns_binarizes_target():
    """
    Test if the normalize_columns function correctly binarizes the target column.
    """
    df = pd.DataFrame({"Age": [50, 60, 70], "Num": [0, 1, 4]})
    out = normalize_columns(df)
    assert out["target"].tolist() == [0, 1, 1], "Target column was not binarized correctly."
    assert "age" in out.columns, "Column names were not normalized correctly."

def test_normalize_columns_requires_target():
    """
    Test if the normalize_columns function raises an error when the target column is missing.
    """
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"age": [50]}))