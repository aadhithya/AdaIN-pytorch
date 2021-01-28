import torch
from os.path import dirname as d
from os.path import abspath
import sys

sys.path.append(d(d(abspath(__file__))))
from model import StyleNet


def test_model_build():
    model = StyleNet()
    x = torch.randn(1, 3, 128, 128)
    out = model(x, x)

    assert (
        out.shape == x.shape
    ), f"Model build failed. out.shape{out.shape} != in.shape{x.shape}"
