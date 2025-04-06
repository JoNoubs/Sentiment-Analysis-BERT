# Purpose: Test the data extraction functionality
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
from src.data_extraction import load_data

def test_load_data(tmp_path):
    # Create a temporary test CSV
    file = tmp_path / "dataset.csv"
    file.write_text("content,score\nGood,5\nBad,1")
    df = load_data(file)
    assert len(df) == 2
    assert list(df.columns) == ['content', 'label']
    assert df['label'].isin([0, 1, 2]).all()

def test_load_data_missing_file():
    # Test error handling for missing file
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent.csv")

def test_load_data_invalid_columns(tmp_path):
    # Test error handling for invalid columns
    file = tmp_path / "dataset.csv"
    file.write_text("wrong,col\nData,here")
    with pytest.raises(ValueError):
        load_data(file)
