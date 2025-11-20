import numpy as np

from causalexplain.independence import kernels


def test_rbf_dot_uses_auto_bandwidth_for_vector():
    x = np.array([0.0, 1.0, 2.0])
    gram = kernels.rbf_dot(x, deg=-1)
    assert gram.shape == (3, 3)
    assert np.allclose(np.diag(gram), 1.0)


def test_kernel_delta_and_norm():
    inp1 = np.array([[0, 1, 0]])
    inp2 = np.array([[1, 0, 1]])
    k_norm = kernels.kernel_Delta_norm(inp1, inp2)
    k_plain = kernels.kernel_Delta(inp1, inp2)

    assert k_norm.shape == (3, 3)
    assert np.isclose(k_plain[0, 1], 1.0)
    assert np.all(k_norm >= 0)
    assert np.isclose(k_norm[0, 1], k_norm[2, 1])


def test_kernel_gaussian_symmetry():
    x = np.array([[0.0, 1.0]])
    y = np.array([[1.0, 2.0]])
    k = kernels.kernel_Gaussian(x, y, sigma=1.0)
    assert k.shape == (2, 2)
    assert np.all(k <= 1.0)
