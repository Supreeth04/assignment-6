import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

class TestModelArchitecture:
    def test_parameter_count(self):
        """Test if model has less than 20k parameters"""
        model = Net()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

    def test_batch_norm_usage(self):
        """Test if model uses Batch Normalization"""
        model = Net()
        has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
        assert has_batch_norm, "Model should use Batch Normalization"

    def test_dropout_usage(self):
        """Test if model uses Dropout"""
        model = Net()
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        assert has_dropout, "Model should use Dropout"

    def test_gap_usage(self):
        """Test if model uses Global Average Pooling"""
        model = Net()
        has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
        fc_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert has_gap or len(fc_layers) == 0, "Model should either use Global Average Pooling or avoid Fully Connected layers"

class TestModelFunctionality:
    def test_model_output_shape(self):
        """Test if model produces correct output shape"""
        model = Net()
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

    def test_forward_pass(self):
        """Test if model forward pass works"""
        model = Net()
        x = torch.randn(1, 1, 28, 28)
        try:
            output = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {str(e)}")

class TestModelPerformance:
    def test_model_size(self):
        """Test model size in MB"""
        model = Net()
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        assert size_all_mb < 1, f"Model size {size_all_mb:.2f}MB exceeds 1MB" 