import numpy as np
import pytest

from causalexplain.estimators.lingam import utils


def test_find_all_paths_and_effects():
    dag = np.array([
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0],
    ])
    paths, effects = utils.find_all_paths(dag, from_index=0, to_index=2)

    assert paths == [[0, 1, 2]]
    assert effects[0] == pytest.approx(0.15)

def test_predict_adaptive_lasso_returns_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    coefs = utils.predict_adaptive_lasso(X, predictors=[0], target=1)

    assert isinstance(coefs, np.ndarray)
    assert coefs.shape == (1,)
