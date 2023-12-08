import pandas as pd
import pytest
from common import Prepup
import plotext as tpl  

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'numeric_column1': [1, 2, 3, 4, 5],
        'numeric_column2': [5, 4, 3, 2, 1],
        'numeric_column3': [1, 2, 3, 4, 5]
        # Add other columns as needed
    })

def test_correlation_n(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.correlation_n()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Correlation Between these Features" in captured.out  # Check for the presence of a title in the output

    # Visual inspection: Manually check the printed output or plot
    print(captured.out)