import numpy as np
import pytest

from causalexplain.estimators.cam import updateScoreMat as update_score_mat_module


def test_update_score_mat_recomputes_allowed_entries(monkeypatch):
    calls = []

    def fake_parallel(row_parents, sel_mat, score_name, X, verbose, node2, pars_score, interv_mat, interv_data, i):
        calls.append((row_parents.copy(), node2, i))
        return 2.0

    monkeypatch.setattr(update_score_mat_module, "computeScoreMatParallel", fake_parallel)

    score_mat = np.zeros((3, 3))
    X = np.ones((4, 3))
    adj = np.zeros((3, 3))
    adj[0, 1] = 1

    updated = update_score_mat_module.updateScoreMat(
        score_mat=score_mat,
        X=X,
        score_name="test",
        i=0,
        j=1,
        score_nodes=np.zeros(3),
        adj=adj,
        verbose=False,
        num_cores=1,
        max_num_parents=2,
        pars_score={},
        interv_mat=None,
        interv_data=False,
    )

    assert updated[2, 1] == 2.0
    assert calls[0][1:] == (1, 2)


def test_update_score_mat_respects_parent_limit(monkeypatch):
    monkeypatch.setattr(
        update_score_mat_module,
        "computeScoreMatParallel",
        lambda *args, **kwargs: pytest.fail("Should not be called when parent limit reached"),
    )

    score_mat = np.zeros((3, 3))
    X = np.ones((4, 3))
    adj = np.zeros((3, 3))
    adj[0, 1] = 1

    updated = update_score_mat_module.updateScoreMat(
        score_mat=score_mat,
        X=X,
        score_name="test",
        i=0,
        j=1,
        score_nodes=np.zeros(3),
        adj=adj,
        verbose=False,
        num_cores=1,
        max_num_parents=0,
        pars_score={},
        interv_mat=None,
        interv_data=False,
    )

    assert updated[2, 1] == -np.inf
