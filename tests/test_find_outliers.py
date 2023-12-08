import pandas as pd
import pytest
import numpy as np
from main.common import Prepup

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'numeric_column1': [1, 2, 3, 4, 5, 100],
        'numeric_column2': [5, 4, 3, 2, 1, 200],
        'numeric_column3': [1, 2, 3, 4, 5, 300]
        # Add other columns as needed
    })

def test_find_outliers(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.find_outliers()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Outliers detected" in captured.out  # Check for the presence of the statement about detected outliers

    # Visual inspection: Manually check the printed output
    print(captured.out)