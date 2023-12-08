import pandas as pd
import pytest
from main.common import Prepup

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': ['A', 'B', 'A']
        
    })

def test_features_available(sample_dataframe):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function you want to test
    result = df.features_available()

    # Convert the Pandas Index to a list
    result_list = result.tolist() if isinstance(result, pd.Index) else result

    # Assert that the result is a list of expected column names
    expected_column_names = ['feature1', 'feature2', 'feature3']
    assert result_list == expected_column_names