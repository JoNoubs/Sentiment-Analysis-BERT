# tests/unit/test_inference.py
# Purpose: Test the inference functionality
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
from src.inference import predict

def test_predict_output_shape():
    # Ensure the model exists (run training if necessary)
    result = predict("This is a test sentence.")
    assert result in [0, 1, 2], f"Expected result to be in [0, 1, 2], got {result}"