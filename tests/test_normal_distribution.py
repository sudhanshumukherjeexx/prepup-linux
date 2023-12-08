import pandas as pd
import pytest
from main.common import Prepup
from termcolor import colored
from pyfiglet import Figlet
from scipy.stats import shapiro

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'numeric_column1': [1, 2, 3, 4, 5],
        'numeric_column2': [5, 4, 3, 2, 1],
        'numeric_column3': [1, 2, 3, 4, 5]
    })

def test_check_normal_distribution(sample_dataframe, capsys):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    df.check_nomral_distrubution()

    # Capture printed output
    captured = capsys.readouterr()

    # Assert that the function runs without errors
    assert "is Normally Distributed" in captured.out or "doesn't have a Normal Distribution" in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)
