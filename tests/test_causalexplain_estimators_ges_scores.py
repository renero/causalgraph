import numpy as np
import pytest

from causalexplain.estimators.ges import scores


def test_gaussobs_initialization_and_method_validation():
    data = np.array([[0.0, 1.0], [1.0, 0.0]])
    score = scores.GaussObsL0Pen(data, lmbda=1.0, method="scatter")
    assert score.p == 2
    assert score.method == "scatter"

    score_raw = scores.GaussObsL0Pen(data, method="raw")
    assert score_raw.method == "raw"

    with pytest.raises(ValueError):
        scores.GaussObsL0Pen(data, method="bad")


def test_gaussobs_local_and_full_score(monkeypatch):
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    score = scores.GaussObsL0Pen(data, method="scatter", lmbda=0.1)

    # stub _mle_local to control outputs
    monkeypatch.setattr(score, "_mle_local", lambda j, parents: (np.zeros(2), 0.5))
    local = score._compute_local_score(0, {1})
    assert local < 0  # likelihood minus penalty

    A = np.array([[0, 1], [0, 0]])
    monkeypatch.setattr(score, "_mle_full", lambda A: (np.zeros_like(A), np.ones(2)))
    total = score.full_score(A)
    assert isinstance(total, float)
