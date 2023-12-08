import pandas as pd
import pytest
from main.common import Prepup


@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'column1': [1, 2, None, 4],
        'column2': ['A', 'B', None, 'D'],
        'column3': [True, False, None, True]
        # Add other columns as needed
    })

def test_shape_data(sample_dataframe):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function
    result = df.shape_data()

    # Assert that the result is a tuple
    assert isinstance(result, tuple)

    # Assert that the shape matches the expected shape
    expected_shape = sample_dataframe.shape
    assert result == expected_shape
