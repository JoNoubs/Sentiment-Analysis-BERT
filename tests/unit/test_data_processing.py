# tests/unit/test_data_processing.py
# Purpose: Test text cleaning functionality
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.data_processing import clean_text

def test_clean_text():
    assert clean_text("Hello!") == "hello"
    assert clean_text("123 ABC!!!") == "123 abc"
    assert clean_text("") == ""
