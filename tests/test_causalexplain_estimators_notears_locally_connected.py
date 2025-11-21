import numpy as np
import torch

from causalexplain.estimators.notears import locally_connected


def test_forward_without_bias_matches_numpy():
    torch.set_default_dtype(torch.double)
    n, d, m1, m2 = 2, 3, 2, 2
    input_array = np.arange(n * d * m1).reshape(n, d, m1).astype(float)
    weight = np.ones((d, m1, m2))

    layer = locally_connected.LocallyConnected(d, m1, m2, bias=False)
    with torch.no_grad():
        layer.weight[:] = torch.from_numpy(weight)

    output = layer(torch.from_numpy(input_array)).detach().numpy()
    expected = np.zeros((n, d, m2))
    for j in range(d):
        expected[:, j, :] = input_array[:, j, :] @ weight[j]

    assert np.allclose(output, expected)


def test_forward_with_bias_adds_offset():
    torch.set_default_dtype(torch.double)
    layer = locally_connected.LocallyConnected(1, 2, 2, bias=True)
    with torch.no_grad():
        layer.weight[:] = torch.tensor([[[1.0, -1.0], [0.5, 0.5]]])
        layer.bias[:] = torch.tensor([[0.2, 0.3]])

    inp = torch.tensor([[[2.0, 1.0]]])  # shape [1, d=1, m1=2]
    output = layer(inp).detach().numpy()

    manual = np.array([[[2.0 * 1.0 + 1.0 * 0.5 + 0.2,
                         2.0 * -1.0 + 1.0 * 0.5 + 0.3]]])
    assert np.allclose(output, manual)
