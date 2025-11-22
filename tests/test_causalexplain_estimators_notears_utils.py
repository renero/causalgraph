import numpy as np
import pytest

from causalexplain.estimators.notears import utils


class FakeGraph:
    def __init__(self, adjacency):
        self.adjacency = np.array(adjacency)

    @classmethod
    def Weighted_Adjacency(cls, adjacency):
        return cls(adjacency)

    @classmethod
    def Adjacency(cls, adjacency):
        return cls(adjacency)

    def is_dag(self):
        # treat absence of cycles in upper triangle only for this stub
        return np.allclose(self.adjacency, np.tril(self.adjacency))

    def get_adjacency(self):
        class Data:
            def __init__(self, arr):
                self.data = arr.tolist()
        return Data(self.adjacency)

    def topological_sorting(self):
        return list(range(self.adjacency.shape[0]))

    def neighbors(self, node, mode=None):
        return list(np.where(self.adjacency[:, node] != 0)[0])


class FakeIG:
    IN = "IN"

    Graph = FakeGraph

    @staticmethod
    def Graph_Erdos_Renyi(*args, **kwargs):
        return FakeGraph(np.zeros((2, 2)))


def test_set_random_seed_reproducible():
    utils.set_random_seed(0)
    first = np.random.rand(3)
    utils.set_random_seed(0)
    second = np.random.rand(3)
    assert np.allclose(first, second)


def test_is_dag_with_fake_igraph(monkeypatch):
    monkeypatch.setattr(utils, "ig", FakeIG)
    dag_matrix = np.array([[0, 0], [1, 0]])
    assert utils.is_dag(dag_matrix)
    cyclic = np.array([[0, 1], [1, 0]])
    assert not utils.is_dag(cyclic)


def test_simulate_parameter_with_ranges():
    utils.set_random_seed(0)
    B = np.array([[0, 1], [0, 0]])
    W = utils.simulate_parameter(B, w_ranges=((-1.0, -0.5), (0.5, 1.0)))
    assert W.shape == B.shape
    assert np.all((W == 0) | (np.abs(W) >= 0.5))


def test_simulate_linear_sem_monkeypatched_ig(monkeypatch):
    monkeypatch.setattr(utils, "ig", FakeIG)
    monkeypatch.setattr(utils, "is_dag", lambda W: True)
    W = np.array([[0.0, 0.0], [1.0, 0.0]])
    X = utils.simulate_linear_sem(W, n=3, sem_type="gauss")
    assert X.shape == (3, 2)
    # second variable should depend on first through W[0,1]
    assert not np.allclose(X[:, 1], 0)


def test_simulate_nonlinear_sem_with_stubbed_graph(monkeypatch):
    monkeypatch.setattr(utils, "ig", FakeIG)
    B = np.array([[0, 0], [1, 0]])
    X = utils.simulate_nonlinear_sem(B, n=2, sem_type="mim", noise_scale=np.ones(2))
    assert X.shape == (2, 2)
