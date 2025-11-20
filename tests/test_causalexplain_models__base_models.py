import numpy as np
import pytest
import torch

from causalexplain.models._base_models import DFF, MDN, MLP


def test_mlp_invalid_activation():
    with pytest.raises(ValueError):
        MLP(
            input_size=2,
            layers_dimensions=[2],
            activation="unsupported",
            batch_size=1,
            lr=0.01,
            loss="mse",
            dropout=0.1,
        )


def test_mlp_invalid_loss():
    with pytest.raises(ValueError):
        MLP(
            input_size=2,
            layers_dimensions=[2],
            activation="relu",
            batch_size=1,
            lr=0.01,
            loss="not-a-loss",
            dropout=0.1,
        )


def test_mlp_forward_and_predict_shape():
    torch.manual_seed(0)
    mlp = MLP(
        input_size=3,
        layers_dimensions=[2],
        activation="relu",
        batch_size=2,
        lr=0.01,
        loss="mse",
        dropout=0.0,
    )
    x = torch.ones((2, 2))
    out = mlp.forward(x)
    assert out.shape == (2, 1)

    np_out = mlp.predict(np.ones((2, 2), dtype=np.float32))
    assert np_out.shape == (2, 1)


def test_dff_invalid_loss():
    with pytest.raises(ValueError):
        DFF(input_size=1, hidden_size=1, batch_size=1, lr=0.01, loss="oops")


@pytest.mark.parametrize("kernel", ["multiscale", "rbf"])
def test_mdn_static_mmd_loss_kernels(kernel):
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[0.5], [1.5]])

    loss_val = MDN.mmd_loss(x, y, kernel)
    assert loss_val >= 0


def test_mdn_common_step_branches(monkeypatch):
    mdn = MDN(
        input_size=1,
        hidden_size=1,
        num_gaussians=1,
        lr=0.01,
        batch_size=1,
        loss_function="loglikelihood",
    )

    # Bypass the real forward call to test the loglikelihood path deterministically.
    def fake_forward(x):
        return (
            torch.tensor([[1.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0]]),
        )

    monkeypatch.setattr(mdn, "forward", fake_forward)
    x = torch.zeros((1, 1))
    y = torch.zeros((1, 1))
    loss = mdn.common_step((x, y))
    assert loss >= 0

    # Now exercise the mmd branch with controlled outputs.
    mdn.loss_fn = "mmd"
    monkeypatch.setattr(mdn, "forward", lambda _x: fake_forward(_x))
    monkeypatch.setattr(mdn, "g_sample", lambda *_, **__: torch.zeros((1, 1)))
    monkeypatch.setattr(mdn, "mmd_loss", lambda *_, **__: torch.tensor(0.5))
    mmd_loss = mdn.common_step((x, y))
    assert mmd_loss == torch.tensor(0.5)


def test_mdn_gaussian_probability_and_sampling():
    pi = torch.tensor([[0.6], [0.4]])
    sigma = torch.tensor([[1.0], [1.0]])
    mu = torch.tensor([[0.0], [0.5]])

    probs = MDN.gaussian_probability(torch.ones((2, 1)), mu, sigma)
    assert probs.shape == mu.shape

    samples = MDN.g_sample(pi, sigma, mu)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == pi.shape[0]
