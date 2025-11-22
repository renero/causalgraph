import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.lingam import bootstrap


class DummyModel(bootstrap.BootstrapMixin):
    def __init__(self):
        self._causal_order = [0, 1]
        self._adjacency_matrix = np.array([[0.0, 0.0], [0.2, 0.0]])

    def fit(self, X):
        X = np.asarray(X)
        self._causal_order = list(range(X.shape[1]))
        self._adjacency_matrix = np.full((X.shape[1], X.shape[1]), 0.1)
        np.fill_diagonal(self._adjacency_matrix, 0.0)
        return self

    def estimate_total_effect(self, X, from_index, to_index):
        return float(from_index + to_index)


def test_bootstrap_validates_and_collects(monkeypatch):
    model = DummyModel()
    data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]])

    result = model.bootstrap(data, n_sampling=2)

    assert result.adjacency_matrices_.shape == (2, 2, 2)
    assert result.total_effects_.shape == (2, 2, 2)

    with pytest.raises(ValueError):
        model.bootstrap(data, n_sampling=0)
    with pytest.raises(ValueError):
        model.bootstrap(data.values, n_sampling="bad")


def test_causal_direction_counts_filters_by_threshold():
    ams = np.array([
        [[[0, 0.5], [0, 0]]],
        [[[0, -0.6], [0.1, 0]]],
    ]).reshape(2, 2, 2)
    res = bootstrap.BootstrapResult(ams, np.zeros_like(ams))

    counts = res.get_causal_direction_counts(min_causal_effect=0.4)
    assert counts["count"][0] == 2
    assert counts["from"][0] == 1 and counts["to"][0] == 0

    counts_split = res.get_causal_direction_counts(
        split_by_causal_effect_sign=True, min_causal_effect=0.4)
    assert "sign" in counts_split and len(counts_split["sign"]) == len(counts_split["from"])

    with pytest.raises(ValueError):
        res.get_causal_direction_counts(n_directions=0)


def test_directed_acyclic_graph_counts_and_probabilities():
    ams = np.array([
        [[0, 0.2], [0, 0]],
        [[0, -0.1], [0.4, 0]],
    ])
    res = bootstrap.BootstrapResult(ams, np.zeros_like(ams))

    dags = res.get_directed_acyclic_graph_counts(min_causal_effect=0.05)
    assert dags["count"][0] >= 1
    assert "from" in dags["dag"][0]

    dag_signs = res.get_directed_acyclic_graph_counts(
        min_causal_effect=0.05, split_by_causal_effect_sign=True)
    assert "sign" in dag_signs["dag"][0]

    probs = res.get_probabilities(min_causal_effect=0.1)
    assert probs.shape == (2, 2)

    ams_wide = np.array([[[0, 1, 0, 0], [0, 0, 0, 0]]])
    res_wide = bootstrap.BootstrapResult(ams_wide, np.zeros_like(ams_wide))
    splitted = res_wide.get_probabilities(min_causal_effect=0.0)
    assert len(splitted) == 2

    with pytest.raises(ValueError):
        res.get_directed_acyclic_graph_counts(n_dags=0)


def test_total_causal_effects_and_paths():
    total_effects = np.zeros((2, 2, 2))
    total_effects[0, 1, 0] = 0.5
    total_effects[1, 1, 0] = 0.8
    ams = np.array([
        [[0, 0.5], [0, 0]],
        [[0, 0.5], [0, 0]],
    ])
    res = bootstrap.BootstrapResult(ams, total_effects)

    te = res.get_total_causal_effects(min_causal_effect=0.0)
    assert te["from"] == [0]
    assert te["to"] == [1]
    assert te["effect"][0] == pytest.approx(0.65)
    assert te["probability"][0] == 1.0

    paths = res.get_paths(from_index=0, to_index=1)
    assert paths["path"][0] == [0, 1]
    assert paths["probability"][0] == 1.0

    with pytest.raises(ValueError):
        res.get_paths(0, 1, min_causal_effect=-1.0)
