import numpy as np
import pytest

from causalexplain.estimators.cam.computeScoreMatParallel import computeScoreMatParallel


def test_compute_score_mat_parallel_semgam_runs(monkeypatch):
    monkeypatch.setattr(
        "causalexplain.estimators.cam.computeScoreMatParallel.train_gam",
        lambda X, y, pars=None, verbose=False: {"residuals": np.array([1.0, 3.0])},
    )
    row_parents = np.array([[0], [1]])
    sel_mat = np.ones((2, 2), dtype=bool)
    X = np.array([[0.0, 1.0], [1.0, 2.0]])

    score = computeScoreMatParallel(
        row_parents=row_parents,
        score_name="SEMGAM",
        X=X,
        sel_mat=sel_mat,
        verbose=False,
        node2=1,
        i=0,
        pars_score={},
        interv_mat=None,
        interv_data=False,
    )

    assert np.isclose(score, 0.0)


def test_compute_score_mat_parallel_prunes_unselected_parent(monkeypatch):
    row_parents = np.array([[0], [1]])
    sel_mat = np.array([[True, False], [True, True]])
    X = np.ones((2, 2))

    score = computeScoreMatParallel(
        row_parents=row_parents,
        score_name="SEMGAM",
        X=X,
        sel_mat=sel_mat,
        verbose=False,
        node2=1,
        i=0,
        pars_score={},
        interv_mat=None,
        interv_data=False,
    )

    assert score == -np.inf


@pytest.mark.parametrize(
    "score_name,exc_type",
    [("SEMSEV", ValueError), ("SEMIND", NotImplementedError)],
)
def test_compute_score_mat_parallel_raises_for_invalid_scores(score_name, exc_type):
    row_parents = np.array([[0], [1]])
    sel_mat = np.ones((2, 2), dtype=bool)
    X = np.ones((2, 2))

    with pytest.raises(exc_type):
        computeScoreMatParallel(
            row_parents=row_parents,
            score_name=score_name,
            X=X,
            sel_mat=sel_mat,
            verbose=False,
            node2=1,
            i=0,
            pars_score={},
            interv_mat=None,
            interv_data=False,
        )
