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
        'numeric_column3': [1, 2, 3, 4, 5]
        # Add other columns as needed
    })

def test_scatter_plot(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.scatter_plot()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Distribution of" in captured.out  # Check for the presence of a title in the output

    # Visual inspection: Manually check the printed output or plots
    print(captured.out)
