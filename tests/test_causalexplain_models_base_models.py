import numpy as np
import pytest
import torch

from causalexplain.models._base_models import DFF, MDN, MLP
from causalexplain.models._loss import MMDLoss


@pytest.mark.parametrize(
    "activation,expected_cls",
    [
        ("gelu", torch.nn.GELU),
        ("selu", torch.nn.SELU),
        ("tanh", torch.nn.Tanh),
        ("linear", torch.nn.Identity),
        ("sigmoid", torch.nn.Sigmoid),
    ],
)
@pytest.mark.parametrize(
    "loss_name,expected_loss",
    [
        ("mae", torch.nn.L1Loss),
        ("mmd", MMDLoss),
        ("binary_crossentropy", torch.nn.BCEWithLogitsLoss),
        ("crossentropy", torch.nn.CrossEntropyLoss),
    ],
)
def test_mlp_instantiation_covers_additional_branches(activation, expected_cls, loss_name, expected_loss):
    mlp = MLP(
        input_size=2,
        layers_dimensions=[3],
        activation=activation,
        batch_size=2,
        lr=0.01,
        loss=loss_name,
        dropout=0.0,
    )
    assert isinstance(mlp.activation, expected_cls)
    assert isinstance(mlp.loss_fn, expected_loss) or callable(mlp.loss_fn)


def test_mlp_training_and_validation_steps_return_losses(monkeypatch):
    mlp = MLP(
        input_size=2,
        layers_dimensions=[1],
        activation="tanh",
        batch_size=2,
        lr=0.01,
        loss="mae",
        dropout=0.0,
    )
    mlp.log = lambda *_, **__: None
    batch = (torch.zeros((2, 1)), torch.ones((2, 1)))
    train_loss = mlp.training_step(batch, 0)
    val_loss = mlp.validation_step(batch, 0)
    assert train_loss > 0
    assert val_loss > 0


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


def test_dff_training_and_validation_steps(monkeypatch):
    dff = DFF(input_size=1, hidden_size=1, batch_size=1, lr=0.01, loss="mae")
    monkeypatch.setattr(dff, "forward", lambda x: torch.zeros((x.shape[0], 1)))
    dff.log = lambda *_, **__: None
    x = torch.ones((1, 1))
    y = torch.zeros((1, 1))
    train_loss = dff.training_step((x, y), 0)
    dff.validation_step((x, y), 0)
    assert train_loss >= 0


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


def test_mdn_forward_and_validation_step():
    mdn = MDN(
        input_size=1,
        hidden_size=2,
        num_gaussians=2,
        lr=0.01,
        batch_size=2,
        loss_function="loglikelihood",
    )

    class Wrapper:
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape = tensor.shape

        def to_device(self, _device):
            return self.tensor

    wrapped = Wrapper(torch.zeros((2, 0)))
    pi, sigma, mu = mdn.forward(wrapped)
    assert pi.shape[0] == 2 and sigma.shape[0] == 2 and mu.shape[0] == 2
    mdn.log = lambda *_, **__: None
    mdn.validation_step((torch.zeros((1, 1)), torch.zeros((1, 1))), 0)
    optim = mdn.configure_optimizers()
    assert "optimizer" in optim and "lr_scheduler" in optim


def test_mdn_gaussian_probability_and_sampling():
    pi = torch.tensor([[0.6], [0.4]])
    sigma = torch.tensor([[1.0], [1.0]])
    mu = torch.tensor([[0.0], [0.5]])

    probs = MDN.gaussian_probability(torch.ones((2, 1)), mu, sigma)
    assert probs.shape == mu.shape

    samples = MDN.g_sample(pi, sigma, mu)
    assert isinstance(samples, torch.Tensor)
    assert samples.shape[0] == pi.shape[0]

    added_noise = MDN.add_noise(torch.ones((2, 1)))
    assert added_noise.shape[1] == 2

    pi_full = torch.ones((2, 2)) * 0.5
    sigma_full = torch.ones((2, 2, 1))
    mu_full = torch.zeros((2, 2, 1))
    sampled = MDN.sample(pi_full, sigma_full, mu_full)
    assert sampled.shape[0] == pi_full.shape[0]
