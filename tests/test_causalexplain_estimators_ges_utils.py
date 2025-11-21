import numpy as np
import pytest

from causalexplain.estimators.ges import utils


def test_neighbors_adj_pa_ch():
    A = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
    ])
    assert utils.neighbors(0, A) == {1}
    assert utils.adj(0, A) == {1, 2}
    assert utils.pa(2, A) == {0, 1}
    assert utils.ch(1, A) == {0, 2}


def test_na_filters_neighbors_adjacent_to_node():
    A = np.array([
        [0, 1],
        [1, 0],
    ])
    assert utils.na(0, 1, A) == {1}
    assert utils.na(1, 0, A) == {0}


def test_is_clique_and_skeleton():
    A = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])
    assert utils.is_clique({0, 1, 2}, A) is False
    assert utils.is_clique({0, 1}, A) is True


def test_is_dag_and_topological_ordering():
    dag = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    assert utils.is_dag(dag)
    order = utils.topological_ordering(dag)
    assert order == [0, 1, 2]

    cyclic = np.array([
        [0, 1],
        [1, 0],
    ])
    assert utils.is_dag(cyclic) is False
    with pytest.raises(ValueError):
        utils.topological_ordering(cyclic)
