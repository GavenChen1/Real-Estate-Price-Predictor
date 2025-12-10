# ------------------------------------------------------------
# Author: Zhihao Wu
# Date: 2025-12-09
# File: test_realestate_pytest.py
# Description:
#   Pytest-based unit tests for RealEstateModel.
#   All comments and docstrings are in English as required.
# ------------------------------------------------------------

import pytest
import os
from model_utils import RealEstateModel

# Load the CSV file located in the same directory as this test file
DATA_FILE = os.path.join(os.path.dirname(__file__), "realtor-data.zip.csv")


@pytest.fixture
def model():
    """
    Create a RealEstateModel instance and load the dataset.
    This fixture is reused across all test cases.
    """
    m = RealEstateModel()
    msg = m.load_data(DATA_FILE, sample_size=None)

    # Ensure dataset is successfully loaded
    assert m.data is not None, "Dataset failed to load."
    assert len(m.data) > 0, "Dataset appears to be empty."
    return m


def test_load_data(model):
    """
    Test that dataset is loaded correctly and key columns exist.
    """
    assert "bed" in model.data.columns
    assert "bath" in model.data.columns
    assert "acre_lot" in model.data.columns


def test_column_integrity(model):
    """
    Verify that all required feature columns are present.
    """
    required_cols = ["bed", "bath", "acre_lot", "house_size",
                     "zip_code", "city", "state"]
    for col in required_cols:
        assert col in model.data.columns, f"Missing required column: {col}"


def test_train_model(model):
    """
    Test that a Random Forest model can be trained successfully.
    """
    msg = model.train_model("Random Forest", use_cv=False)
    assert model.model is not None, "Model training failed."


def test_evaluate_model(model):
    """
    Test that evaluate_model returns valid metrics.
    """
    model.train_model("Random Forest", use_cv=False)
    metrics = model.evaluate_model()

    assert "MSE" in metrics, "Missing MSE metric."
    assert "R2" in metrics, "Missing R2 metric."
    assert metrics["MSE"] >= 0, "Invalid MSE value returned."


def test_predict(model):
    """
    Test that the prediction function returns valid results.
    """
    model.train_model("Random Forest", use_cv=False)

    row = model.data.iloc[0]

    price, loc = model.predict(
        bed=row["bed"],
        bath=row["bath"],
        acre_lot=row["acre_lot"],
        house_size=row["house_size"],
        zip_code=row["zip_code"],
        city=row["city"],
        state=row["state"],
    )

    assert isinstance(price, float), "Predicted price is not a float."
    assert price > 0, "Predicted price is not positive."
    assert isinstance(loc, dict), "Location info should be a dictionary."
