import networkx as nx
import numpy as np
import pytest

import causalexplain.estimators.pc.dag as dag_module
from causalexplain.estimators.pc.dag import DAG


def test_cycle_detection_raises():
    with pytest.raises(ValueError):
        DAG([("A", "B"), ("B", "A")])


def test_add_nodes_from_weight_mismatch():
    dag = DAG()
    with pytest.raises(ValueError):
        dag.add_nodes_from(["A", "B"], weights=[1])


def test_moralize_and_immoralities():
    dag = DAG([("A", "C"), ("B", "C")])
    immoralities = dag.get_immoralities()
    assert ("A", "B") in {tuple(sorted(x)) for x in immoralities}

    dag_module.UndirectedGraph = nx.Graph
    moral = dag.moralize()
    assert moral.has_edge("A", "B")


def test_getters_and_do_operation():
    dag = DAG([("A", "B"), ("B", "C")])
    dag.add_nodes_from(["A", "B", "C"], weights=[1, 2, 3])

    assert dag.get_parents("C") == ["B"]
    assert set(dag.get_children("A")) == {"B"}
    assert dag.get_leaves() == ["C"]

    modified = dag.do("B")
    assert not modified.has_edge("A", "B")


def test_active_trail_and_dconnection():
    dag = DAG([("A", "B"), ("B", "C")])
    trails = dag.active_trail_nodes("A")
    assert set(trails["A"]) == {"A", "B", "C"}

    blocked = dag.active_trail_nodes("A", observed=["B"])
    assert "C" not in blocked["A"]
    assert dag.is_dconnected("A", "C")
    assert not dag.is_dconnected("A", "C", observed=["B"])


def test_minimal_dseparator_and_ancestors():
    dag = DAG([("A", "C"), ("B", "C"), ("C", "D")])
    with pytest.raises(ValueError):
        dag.minimal_dseparator("A", "C")

    separator = dag.minimal_dseparator("A", "D")
    assert separator == {"C"}

    with pytest.raises(ValueError):
        dag._get_ancestors_of("X")


def test_get_random_and_roots():
    dag = DAG.get_random(n_nodes=3, edge_prob=0.0, latents=True)
    assert len(list(dag.nodes())) == 3
    assert set(dag.get_roots()) == set(dag.nodes())
