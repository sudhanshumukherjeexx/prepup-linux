import pandas as pd
import pytest
from main.common import Prepup
import plotext as tpl  


@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'numeric_column1': [1, 2, 3, 4, 5],
        'numeric_column2': [5, 4, 3, 2, 1],
        'categorical_column': ['A', 'B', 'A', 'B', 'A']
        
    })

def test_plot_histogram(sample_dataframe, capsys):
    # Class instance
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.plot_histogram()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "numeric_column1" in captured.out  # Check for the presence of a column name in the output

    # Visual inspection: Manually check the printed output or plot
    print(captured.out)