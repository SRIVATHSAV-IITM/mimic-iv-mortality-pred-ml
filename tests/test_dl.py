from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train_dl import TabularMLP  # noqa: E402


def test_tabular_mlp_output_shape() -> None:
    model = TabularMLP(input_features=12, hidden_features=32)
    output = model(torch.zeros((5, 12)))
    assert tuple(output.shape) == (5,)
