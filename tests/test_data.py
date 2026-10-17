import pandas as pd
import pytest
from heart_disease.data import normalize_columns


def test_normalize_columns_binarizes_target():
    df = pd.DataFrame({"Age": [50, 60, 70], "Num": [0, 1, 4]})
    out = normalize_columns(df)
    assert out["target"].tolist() == [0, 1, 1]
    assert "age" in out.columns


def test_normalize_columns_requires_target():
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"age": [50]}))
