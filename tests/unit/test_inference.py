# tests/unit/test_inference.py
# Purpose: Test the inference functionality
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pytest
import torch
from src.inference import predict

def test_inference(mocker):
    # Mock a model for testing
    model = torch.nn.Linear(10, 3)  # Simple mock model
    torch.save(model.state_dict(), "./models/pytorch_model.bin")
    result = predict("test text", model_path="./models")
    assert result in [0, 1, 2]
