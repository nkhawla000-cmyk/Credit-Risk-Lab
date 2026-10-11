"""Small local tests: independent of the original UCI download."""
import pandas as pd
import pytest

from src.data import FEATURES, load_data, MODEL_FEATURES, SENSITIVE


def test_data_schema_and_class_mapping(tmp_path):
    row = ["A11", 24, "A34", "A43", 2000, "A61", "A72", 2, "A93", "A101", 4,
           "A121", 30, "A143", "A152", 1, "A173", 1, "A191", "A201"]
    path = tmp_path / "credit.data"
    pd.DataFrame([row + [1], row + [2]]).to_csv(path, sep=" ", header=False, index=False)
    data = load_data(path)
    assert data["default"].tolist() == [0, 1]
    assert len(FEATURES) == 20
    assert len(MODEL_FEATURES) == 17
    assert set(MODEL_FEATURES).isdisjoint(SENSITIVE)


def test_invalid_label_fails(tmp_path):
    row = ["A11", 24, "A34", "A43", 2000, "A61", "A72", 2, "A93", "A101", 4,
           "A121", 30, "A143", "A152", 1, "A173", 1, "A191", "A201", 3]
    path = tmp_path / "invalid.data"
    pd.DataFrame([row]).to_csv(path, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="classes attendues"):
        load_data(path)
