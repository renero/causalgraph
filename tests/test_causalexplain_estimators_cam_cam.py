import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.cam.cam import CAM


def test_cam_requires_fit_before_predict():
    model = CAM(name="test")
    with pytest.raises(ValueError):
        model.predict()


def test_cam_fit_and_predict_flow(monkeypatch):
    def fake_compute_score_mat(*args, **kwargs):
        return {
            "scoreMat": np.array([[-np.inf, 0.5], [0.1, -np.inf]]),
            "rowParents": np.array([[0], [1]]),
            "scoreEmptyNodes": np.zeros(2),
        }

    monkeypatch.setattr("causalexplain.estimators.cam.cam.computeScoreMat", fake_compute_score_mat)
    monkeypatch.setattr(
        "causalexplain.estimators.cam.cam.updateScoreMat",
        lambda score_mat, *args, **kwargs: np.full_like(score_mat, -np.inf),
    )
    monkeypatch.setattr(
        "causalexplain.estimators.cam.cam.pruning",
        lambda X, G, prune_method=None, prune_method_pars=None, verbose=False: G,
    )
    fake_graph = object()
    monkeypatch.setattr(
        "causalexplain.estimators.cam.cam.utils.graph_from_adjacency",
        lambda adj, names: fake_graph,
    )
    monkeypatch.setattr(
        "causalexplain.estimators.cam.cam.evaluate_graph",
        lambda ref_graph, dag, feature_names: {"dummy": 1},
    )

    data = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], columns=["a", "b"])
    model = CAM(name="test_cam", pruning=True, verbose=False)

    fitted = model.fit(data)

    assert fitted.is_fitted_
    assert fitted.adj[0, 1] == 1
    assert fitted.adj[1, 0] == 0

    dag = fitted.predict()
    assert dag is fake_graph
