import numpy as np

from causalexplain.estimators.cam import train_lasso


def test_train_lasso_fits_and_produces_residuals():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=20)

    result = train_lasso.train_lasso(X, y)

    assert set(result) == {"Yfit", "residuals", "model"}
    assert result["Yfit"].shape == y.shape
    assert result["residuals"].shape == y.shape
    assert getattr(result["model"], "alpha_", None) or getattr(result["model"], "alpha") is not None
