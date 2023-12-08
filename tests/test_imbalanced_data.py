import pandas as pd
import pytest
from main.common import Prepup
import plotext as tpl  

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'feature1': [1, 2, 1, 1, 2, 2, 1, 1, 1],
        'feature2': [0, 1, 1, 0, 1, 1, 1, 0, 0],
        'target_variable': [1, 0, 1, 0, 1, 1, 0, 0, 0]
        # Add other columns as needed
    })

def test_imbalanced_dataset(sample_dataframe, capsys, monkeypatch):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Mock user input for the target variable
    monkeypatch.setattr('builtins.input', lambda _: 'target_variable')

    # Call the function
    df.imbalanced_dataset()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "Target Variable Distribution" in captured.out  # Check for the presence of the title in the output

    # Visual inspection: Manually check the printed or plotted output
    print(captured.out)
