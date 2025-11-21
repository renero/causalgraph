import numpy as np
import pytest

from causalexplain.estimators.notears import notears


def test_notears_standard_uses_custom_loss(monkeypatch):
    monkeypatch.setattr(notears.scipy.linalg, "expm",
                        lambda mat: np.eye(mat.shape[0]))

    calls = {"loss": 0, "grad": 0}

    def fake_minimize(func, w_flat, args=None, jac=None, method=None, options=None):
        class Res:
            def __init__(self, x):
                self.x = x
        return Res(w_flat)

    monkeypatch.setattr(notears.scipy.optimize, "minimize", fake_minimize)

    def custom_loss(W, data, cov, d, n):
        calls["loss"] += 1
        return 0.0

    def custom_grad(W, data, cov, d, n):
        calls["grad"] += 1
        return np.zeros_like(W)

    algo = notears.NOTEARS(name="test", loss=custom_loss, loss_grad=custom_grad)
    data = np.zeros((2, 2))
    out = algo.notears_standard(data, return_all_progress=False)

    assert calls["loss"] >= 1 and calls["grad"] >= 1
    assert out["W"].shape == (2, 2)
    assert out["h"] == 0.0


def test_fit_and_predict_flow(monkeypatch):
    model_output = {"W": np.array([[0.2, 0.0], [0.0, 0.0]])}

    def fake_notears_standard(self, data, return_all_progress=False):
        return model_output

    monkeypatch.setattr(notears.NOTEARS, "notears_standard", fake_notears_standard)
    monkeypatch.setattr(notears.utils, "graph_to_adjacency",
                        lambda ref_graph, labels: np.array([[0, 1], [0, 0]]))

    graph_inputs = {}
    monkeypatch.setattr(
        notears.utils,
        "graph_from_adjacency",
        lambda adjacency, node_labels: graph_inputs.update(
            {"adjacency": adjacency, "labels": node_labels}
        ) or "graph_obj")

    monkeypatch.setattr(notears, "evaluate_graph",
                        lambda ref_graph, dag: {"ref": ref_graph, "dag": dag})

    algo = notears.NOTEARS(name="pipeline")
    import pandas as pd

    train = pd.DataFrame([[0.0, 0.0]], columns=["a", "b"])
    ref_graph = "ref"
    algo.fit(train)
    algo.predict(ref_graph=ref_graph, threshold=0.1)

    assert np.array_equal(graph_inputs["adjacency"], np.array([[1, 0], [0, 0]]))
    assert graph_inputs["labels"] == ["a", "b"]
    assert algo.metrics == {"ref": ref_graph, "dag": "graph_obj"}
    assert algo.dag == "graph_obj"


def test_fit_predict_returns_graph(monkeypatch):
    monkeypatch.setattr(notears.NOTEARS, "fit",
                        lambda self, X: setattr(self, "model", {"W": np.zeros((1, 1))}) or self)
    monkeypatch.setattr(notears.NOTEARS, "predict",
                        lambda self, ref_graph=None, threshold=0.1: setattr(self, "dag", "dag_obj") or self)

    algo = notears.NOTEARS(name="fit-predict")
    dag = algo.fit_predict(train=np.array([[0.0]]), test=None, ref_graph=None)
    assert dag == "dag_obj"
