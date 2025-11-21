import numpy as np
import torch
import scipy.linalg as slin

from causalexplain.estimators.notears.trace_expm import trace_expm


def test_trace_expm_forward_and_backward():
    torch.set_default_dtype(torch.double)
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    result = trace_expm(matrix)
    expected = slin.expm(matrix.detach().numpy()).trace()
    assert np.isclose(result.item(), expected)

    scalar = 0.5 * result * result
    scalar.backward()
    expected_grad = result.item() * slin.expm(matrix.detach().numpy()).T
    assert np.allclose(matrix.grad.detach().numpy(), expected_grad)
