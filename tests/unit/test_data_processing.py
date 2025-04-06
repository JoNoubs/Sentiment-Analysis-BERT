# Purpose: Test text cleaning, tokenization, and data splitting
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pandas as pd
from src.data_processing import clean_text, tokenize_data, split_data

def test_clean_text():
    assert clean_text("Hello!") == "hello"
    assert clean_text("123 ABC!!!") == "123 abc"

def test_tokenize_data():
    df = pd.DataFrame({"content": ["good movie"], "label": [2]})
    encodings, labels = tokenize_data(df)
    assert 'input_ids' in encodings
    assert len(labels) == 1
    assert len(encodings['input_ids']) == 1

def test_split_data():
    df = pd.DataFrame({"content": ["a", "b", "c", "d", "e"], "label": [0, 1, 0, 1, 2]})
    train_df, val_df = split_data(df)
    assert len(train_df) == 4
    assert len(val_df) == 1
