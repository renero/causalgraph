import numpy as np

from causalexplain.explainability.regression_quality import RegQuality


def test_predict_intersects_gamma_and_mad():
    scores = [1.0, 1.0, 10.0]
    result = RegQuality.predict(scores, gamma_shape=1, gamma_scale=1, threshold=0.05)
    assert result == {2}


def test_gamma_and_mad_helpers_handle_small_sets():
    scores = np.array([0.1, 0.2, 0.3])
    gamma_idx = RegQuality._gamma_criteria(scores, gamma_shape=1, gamma_scale=1, threshold=10)
    assert gamma_idx == {0, 1, 2}

    mad_idx = RegQuality._mad_criteria(scores, verbose=True)
    assert mad_idx == set()
