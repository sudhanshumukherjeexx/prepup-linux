import pytest
import main.common  
import pandas as pd
import os
from main.common import Prepup



@pytest.fixture
def f_scaling():
   df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]}) 
   return Prepup(df)

def test_feature_scaling_1(f_scaling, monkeypatch, tmpdir, capsys):
    outputs = ["None", "1", "."]
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))

    f_scaling.feature_scaling()

    # Capture printed output
    captured = capsys.readouterr()

    # Print the captured output for inspection
    print(captured.out)

    # Assert that the function runs without errors
    assert "Feature Normalized and saved successfully" in captured.out

def test_feature_scaling_2(f_scaling, monkeypatch, tmpdir, capsys):
    outputs = ["None", "2", "."]
    monkeypatch.setattr('builtins.input', lambda x: outputs.pop(0))

    f_scaling.feature_scaling()

    # Capture printed output
    captured = capsys.readouterr()

    # Print the captured output for inspection
    print(captured.out)

    # Assert that the function runs without errors
    assert "Feature Normalized and saved successfully" in captured.out  