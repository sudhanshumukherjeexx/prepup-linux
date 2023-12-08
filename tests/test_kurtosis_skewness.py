import pandas as pd
import pytest
from main.common import Prepup
from scipy.stats import skew, kurtosis

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'feature1': [1, 2, 1, 1, 2, 2, 1, 1, 1],
        'feature2': [0, 1, 1, 0, 1, 1, 1, 0, 0],
        'numeric_column1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'numeric_column2': [9, 8, 7, 6, 5, 4, 3, 2, 1]
        # Add other columns as needed
    })

def test_skewness(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.skewness()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Skewness present in the data" in captured.out  # Check for the presence of the statement in the output

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_kurtosis(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.kurtosis()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Kurtosis present in the data" in captured.out  # Check for the presence of the statement in the output

    # Visual inspection: Manually check the printed output
    print(captured.out)
