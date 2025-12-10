# Author: Zhihao Wu
# Date: 2025-12-09
# Description:
#   Pytest-based unit tests for RealEstateModel.

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

