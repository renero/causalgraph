import numpy as np

from causalexplain.estimators.cam import train_LMboost


def test_train_lmboost_returns_expected_keys():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    result = train_LMboost.train_LMboost(X, y)

    assert set(result) == {"Yfit", "residuals", "model"}
    assert result["Yfit"].shape == y.shape
    centered_y = y - y.mean()
    assert np.allclose(result["Yfit"] + result["residuals"], centered_y, atol=1e-3)
