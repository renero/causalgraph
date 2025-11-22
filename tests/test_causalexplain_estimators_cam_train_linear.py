import numpy as np

from causalexplain.estimators.cam import train_linear


def test_train_linear_predictions_and_residuals():
    X = np.array([[1.0, 2.0], [2.0, 0.0], [3.0, 4.0]])
    y = np.array([3.0, 1.0, 7.0])

    result = train_linear.train_linear(X, y)

    assert set(result) == {"Yfit", "residuals", "model"}
    assert result["Yfit"].shape == (3, 1)
    assert result["residuals"].shape == (3, 1)
    assert np.allclose(result["residuals"].flatten() + result["Yfit"].flatten(), y)
