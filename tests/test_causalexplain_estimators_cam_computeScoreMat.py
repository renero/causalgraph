import numpy as np
import pytest

from causalexplain.estimators.cam import computeScoreMat as compute_score_mat_module


def test_compute_score_mat_invokes_parallel_and_shapes(monkeypatch):
    calls = []

    def fake_parallel(row_parents, score_name, X, sel_mat, verbose, node2, i, pars_score, interv_mat, interv_data):
        calls.append((tuple(row_parents.flatten()), score_name, node2, i, interv_data))
        return float(node2 * 10 + i)

    monkeypatch.setattr(compute_score_mat_module, "computeScoreMatParallel", fake_parallel)

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    sel_mat = np.ones((2, 2), dtype=bool)
    result = compute_score_mat_module.computeScoreMat(
        X=X,
        score_name="test",
        num_parents=1,
        verbose=False,
        num_cores=1,
        sel_mat=sel_mat,
        pars_score={},
        interv_mat=None,
        interv_data=False,
    )

    assert set(result) == {"scoreMat", "rowParents", "scoreEmptyNodes"}
    assert result["scoreMat"].shape == (2, 2)
    assert calls[0][2:] == (0, 0, False)
    assert np.array_equal(result["scoreMat"], np.array([[0.0, 1.0], [10.0, 11.0]]))


def test_compute_score_mat_uses_interventions(monkeypatch):
    monkeypatch.setattr(
        compute_score_mat_module,
        "computeScoreMatParallel",
        lambda *args, **kwargs: 1.0,
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    interv_mat = np.array([[True, False], [False, False], [False, False]])
    sel_mat = np.ones((2, 2), dtype=bool)

    result = compute_score_mat_module.computeScoreMat(
        X=X,
        score_name="test",
        num_parents=1,
        verbose=False,
        num_cores=1,
        sel_mat=sel_mat,
        pars_score={},
        interv_mat=interv_mat,
        interv_data=True,
    )

    assert result["scoreEmptyNodes"][0] == pytest.approx(-np.log(1))
    assert result["scoreMat"][0, 1] == pytest.approx(1 - (-np.log(np.var(X[:, 1]))))
