import types
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.explainability import perm_importance as pimod


class DummyProgBar:
    def start_subtask(self, *args, **kwargs):
        return types.SimpleNamespace(update_subtask=lambda *a, **k: None, remove=lambda *a, **k: None)


class DummyEstimator:
    def __init__(self, name):
        self.name = name


class DummyModels:
    def __init__(self):
        self.regressor = {"a": DummyEstimator("a"), "b": DummyEstimator("b")}


def _patch_progbar(monkeypatch):
    monkeypatch.setattr(pimod, "ProgBar", lambda *a, **k: DummyProgBar())


def test_fit_sklearn_inserts_zero_for_correlated(monkeypatch):
    _patch_progbar(monkeypatch)
    monkeypatch.setattr(
        pimod.Hierarchies, "compute_correlation_matrix", staticmethod(lambda X: pd.DataFrame(np.eye(X.shape[1]), columns=X.columns, index=X.columns)),
    )
    monkeypatch.setattr(
        pimod.Hierarchies, "compute_correlated_features", staticmethod(lambda corr, th, names, verbose=False: {"a": ["b"], "b": ["a"]}),
    )
    monkeypatch.setattr(
        pimod,
        "permutation_importance",
        lambda estimator, X, y, n_repeats, random_state: {"importances_mean": np.array([]), "importances_std": np.array([])},
    )

    models = DummyModels()
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 0.4, 0.3]})
    pi = pimod.PermutationImportance(models, correlation_th=0.8, mean_pi_percentile=0.5, prog_bar=False, verbose=False)
    pi.fit(df)

    assert pi.is_fitted_
    assert pi.pi["a"]["importances_mean"][0] == 0.0  # zero inserted for correlated 'b'
    assert pi.pi["b"]["importances_mean"][0] == 0.0  # zero inserted for correlated 'a'
    assert pi.mean_pi_threshold >= 0.0


def test_predict_sklearn_builds_graph(monkeypatch):
    _patch_progbar(monkeypatch)

    monkeypatch.setattr(
        pimod,
        "permutation_importance",
        lambda estimator, X, y, n_repeats, random_state: {"importances_mean": np.array([0.2]), "importances_std": np.array([0.05])},
    )
    monkeypatch.setattr(pimod.Hierarchies, "compute_correlation_matrix", staticmethod(lambda X: pd.DataFrame(np.eye(X.shape[1]), columns=X.columns, index=X.columns)))
    monkeypatch.setattr(pimod.Hierarchies, "compute_correlated_features", staticmethod(lambda corr, th, names, verbose=False: {}))
    monkeypatch.setattr(pimod, "select_features", lambda values, feature_names, **kwargs: feature_names[:1])

    def fake_digraph(X, feature_names, models, connections, root_causes, reciprocity=True, anm_iterations=10, verbose=False):
        edges = [(target, conn) for target, conns in connections.items() for conn in conns]
        g = nx.DiGraph()
        g.add_edges_from(edges)
        return g

    monkeypatch.setattr(pimod.utils, "digraph_from_connected_features", fake_digraph)

    models = DummyModels()
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    pi = pimod.PermutationImportance(models, prog_bar=False, verbose=False)
    pi.fit(df)
    graph = pi.predict(df)

    assert isinstance(graph, nx.DiGraph)
    assert len(graph.edges()) == 2  # each feature connects to one other
