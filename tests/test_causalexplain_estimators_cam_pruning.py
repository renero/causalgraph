import numpy as np

from causalexplain.estimators.cam import pruning as pruning_module


def test_pruning_keeps_selected_parents(monkeypatch):
    def fake_prune_method(X, pars=None, verbose=False, k=None):
        return np.array([True])

    X = np.array([[0.0, 1.0], [1.0, 2.0]])
    G = np.array([[0, 1], [0, 0]])

    final = pruning_module.pruning(X, G, prune_method=fake_prune_method)

    assert np.array_equal(final, np.array([[0, 1], [0, 0]]))


def test_pruning_handles_nodes_without_parents():
    X = np.zeros((2, 2))
    G = np.zeros((2, 2))

    final = pruning_module.pruning(X, G)

    assert np.array_equal(final, np.zeros_like(G))
