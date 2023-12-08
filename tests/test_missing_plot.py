import pandas as pd
import pytest
from main.common import Prepup
import plotext as tpl  


@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'column1': [1, 2, None, 4],
        'column2': ['A', 'B', None, 'D'],
        'column3': [True, False, None, True]
        # Add other columns as needed
    })

def test_missing_plot(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.missing_plot()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "No Missing Value Found" in captured.out or "Features" in captured.out

    # Visual inspection: Manually check the printed output or plot
    print(captured.out)
