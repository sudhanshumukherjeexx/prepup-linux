import pandas as pd
import pytest
from main.common import Prepup

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    return pd.DataFrame({
        'numeric_feature': [1, 2, 3],
        'categorical_feature': ['A', 'B', 'A'],
        'boolean_feature': [True, False, True]
    })

def test_dtype_features(sample_dataframe):
    # Class instance with the sample DataFrame
    df = Prepup(dataframe=sample_dataframe)

    # Call the function you want to test
    result = df.dtype_features()

    # Assert that the result is a Pandas Series
    assert isinstance(result, pd.Series)

    # Assert that the data types match the expected data types
    expected_data_types = pd.Series({
        'numeric_feature': int,
        'categorical_feature': object,
        'boolean_feature': bool
        # Add other columns and their expected data types as needed
    })

    assert result.equals(expected_data_types)
