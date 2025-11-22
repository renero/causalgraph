import numpy as np

from causalexplain.estimators.notears import main


def test_run_dispatches_to_variant(monkeypatch):
    def variant(data, loss, loss_grad, **kwargs):
        return {"called": True, "kwargs": kwargs}

    result = main.run(variant, data="data", loss="loss", loss_grad="grad", extra=1)
    assert result == {"called": True, "kwargs": {"extra": 1}}


def test_notears_standard_returns_solution(monkeypatch):
    monkeypatch.setattr(main.scipy.linalg, "expm",
                        lambda mat: np.eye(mat.shape[0]))

    def fake_minimize(func, w_flat, args=None, jac=None, method=None, options=None):
        class Res:
            def __init__(self, x):
                self.x = x
        return Res(w_flat)

    monkeypatch.setattr(main.scipy.optimize, "minimize", fake_minimize)

    data = np.array([[0.0, 0.0]])

    def simple_loss(W, data, cov, d, n):
        return float(np.sum(W ** 2))

    def simple_grad(W, data, cov, d, n):
        return 2 * W

    result = main.notears_standard(
        data=data,
        loss=simple_loss,
        loss_grad=simple_grad,
        max_iter=1,
        verbose=False,
    )
    assert "W" in result and "h" in result and "loss" in result
    assert result["W"].shape == (2, 2)
    assert result["h"] == 0.0


def test_notears_standard_can_return_progress(monkeypatch):
    monkeypatch.setattr(main.scipy.linalg, "expm",
                        lambda mat: np.eye(mat.shape[0]))
    monkeypatch.setattr(main.scipy.optimize, "minimize",
                        lambda func, w_flat, args=None, jac=None, method=None, options=None: type("Res", (), {"x": w_flat}))

    data = np.array([[0.0, 0.0]])
    progress = main.notears_standard(
        data=data,
        loss=lambda W, data, cov, d, n: 0.0,
        loss_grad=lambda W, data, cov, d, n: np.zeros_like(W),
        max_iter=1,
        output_all_progress=True)

    assert isinstance(progress, list)
    assert progress[0]["W"].shape == (2, 2)
