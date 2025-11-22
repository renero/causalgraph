import numpy as np

from causalexplain.estimators.cam import train_GAMboost


def test_train_gamboost_builds_spline_and_model():
    X = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel()

    result = train_GAMboost.train_GAMboost(X, y)

    assert set(result) == {"Yfit", "residuals", "model"}
    assert result["Yfit"].shape == y.shape
    assert result["residuals"].shape == y.shape
