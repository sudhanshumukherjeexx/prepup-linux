import pytest
from main.common import Prepup  
import pandas as pd
import os

@pytest.fixture
def sample_dataframe():
    # sample DataFrame for testing
    df = pd.DataFrame({
        'feature1': [1, 2, None, 1, 2, 2, None, 1, 1],
        'feature2': [0, 1, 1, None, 1, 1, 1, 0, 0],
        'numeric_column1': [1, 2, None, 4, 5, 6, 7, None, 9],
        'numeric_column2': [9, 8, 7, 6, None, 4, 3, 2, 1]
    })
    return Prepup(df)


def test_handle_missing_values_1(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["1", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))

    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)


def test_handle_missing_values_2(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["2", "100" ,"."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)


def test_handle_missing_values_3(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["3", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_handle_missing_values_4(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["4", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_handle_missing_values_5(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["5", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_handle_missing_values_6(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["6", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_handle_missing_values_7(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["7", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)

def test_handle_missing_values_8(sample_dataframe, monkeypatch,tmpdir,  capsys):
    outputs = ["8", "."]
    # Mock user input for the choice
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))


    # Call the function
    sample_dataframe.handle_missing_values()

    # Capture printed output
    captured = capsys.readouterr()
    print(captured)

    # Assert that the function runs without errors
    assert "Missing Data Imputed and saved succesfully" in captured.out
    assert "Done..." in captured.out

    # Visual inspection: Manually check the printed output
    print(captured.out)