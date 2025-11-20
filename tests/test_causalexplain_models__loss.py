import torch

from causalexplain.models._loss import MMDLoss, RBF


def test_rbf_bandwidth_and_kernel_values():
    rbf = RBF()
    X = torch.tensor([[0.0], [1.0]])
    L2 = torch.cdist(X, X) ** 2

    # Auto-computed bandwidth from pairwise distances.
    assert rbf.get_bandwidth(L2) == 1.0

    # Fixed bandwidth should bypass the automatic path.
    fixed = RBF(bandwidth=0.5)
    assert fixed.get_bandwidth(L2) == 0.5

    kernel_matrix = rbf(X)
    assert kernel_matrix.shape == (2, 2)
    assert torch.all(kernel_matrix > 0)


def test_mmd_loss_balanced_samples_zero_gap():
    X = torch.tensor([[0.0], [1.0]])
    Y = torch.tensor([[0.0], [1.0]])

    loss_fn = MMDLoss()
    loss = loss_fn(X, Y)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
