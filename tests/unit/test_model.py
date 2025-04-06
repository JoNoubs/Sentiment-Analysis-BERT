# Purpose: Test the model training functionality
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import pandas as pd
from src.model import train_model

@pytest.fixture
def mock_data(mocker):
    mocker.patch('src.data_extraction.load_data', return_value=pd.DataFrame({'content': ['test'], 'label': [1]}))
    mocker.patch('src.data_processing.tokenize_data', return_value=({'input_ids': [[101, 102]], 'attention_mask': [[1, 1]]}, [1]))
    mocker.patch('src.data_processing.split_data', return_value=(pd.DataFrame({'content': ['test'], 'label': [1]}), pd.DataFrame()))

def test_model_training(mock_data):
    train_model()  # Should run without errors
