import networkx as nx
import pandas as pd
import pytest

from causalexplain.estimators.pc.pc import PC
from causalexplain.estimators.pc import pc as pc_module
from causalexplain.estimators.pc.pdag import PDAG


def test_fit_validates_variant_and_ci_test():
    pc = PC(name="demo", variant="invalid")
    with pytest.raises(ValueError):
        pc.fit(pd.DataFrame({"A": [1]}))

    pc = PC(name="demo", ci_test="bad")
    with pytest.raises(ValueError):
        pc.fit(pd.DataFrame({"A": [1]}))


def test_fit_returns_skeleton_and_dag(monkeypatch):
    data = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
    pc = PC(name="demo", variant="stable", return_type="skeleton")

    skeleton = nx.Graph()
    skeleton.add_edge("A", "B")
    monkeypatch.setattr(pc, "build_skeleton", lambda **kwargs: (skeleton, {}))

    skel, sep_sets = pc.fit(data)
    assert skel is skeleton and sep_sets == {}

    pc2 = PC(name="demo", return_type="dag")
    monkeypatch.setattr(pc2, "build_skeleton", lambda **kwargs: (skeleton, {frozenset({"A", "B"}): set()}))

    class DummyPDAG(PDAG):
        def __init__(self):
            super().__init__(directed_ebunch=[("A", "B")], undirected_ebunch=[])

        def to_dag(self):
            return nx.DiGraph([("A", "B")])

    monkeypatch.setattr(pc_module.PC, "skeleton_to_pdag", staticmethod(lambda sk, sep: DummyPDAG()))
    dag = pc2.fit(data)
    assert isinstance(dag, nx.DiGraph)
    assert pc2.is_fitted_


def test_fit_predict_calls_evaluate_graph(monkeypatch):
    data = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
    pc = PC(name="demo", return_type="dag")

    skeleton = nx.Graph()
    skeleton.add_edge("A", "B")
    monkeypatch.setattr(pc, "build_skeleton", lambda **kwargs: (skeleton, {frozenset({"A", "B"}): set()}))
    monkeypatch.setattr(pc_module.PC, "skeleton_to_pdag", staticmethod(lambda sk, sep: PDAG(directed_ebunch=[("A", "B")], undirected_ebunch=[])))

    metrics_called = {}

    def fake_evaluate_graph(ref, dag, feature_names=None):
        metrics_called["called"] = True
        return "metrics"

    monkeypatch.setattr(pc_module, "evaluate_graph", fake_evaluate_graph)

    ref_graph = nx.DiGraph()
    ref_graph.add_node("A")
    result = pc.fit_predict(data, data, ref_graph=ref_graph)
    assert isinstance(result, nx.DiGraph)
    assert metrics_called["called"]


def test_skeleton_to_pdag_orients_v_structure():
    skeleton = nx.Graph()
    skeleton.add_edge("X", "Z")
    skeleton.add_edge("Y", "Z")
    separating_sets = {frozenset({"X", "Y"}): set()}

    pdag = PC.skeleton_to_pdag(skeleton, separating_sets)
    assert ("X", "Z") in pdag.directed_edges
    assert ("Y", "Z") in pdag.directed_edges
    assert ("Z", "X") not in pdag.directed_edges


def test_fit_invalid_return_type(monkeypatch):
    data = pd.DataFrame({"A": [1, 2], "B": [2, 3]})
    pc = PC(name="demo", return_type="unknown")
    skeleton = nx.Graph()
    skeleton.add_edge("A", "B")
    monkeypatch.setattr(pc, "build_skeleton", lambda **kwargs: (skeleton, {frozenset({"A", "B"}): set()}))
    monkeypatch.setattr(pc_module.PC, "skeleton_to_pdag", staticmethod(lambda sk, sep: PDAG(directed_ebunch=[("A", "B")], undirected_ebunch=[])))

    with pytest.raises(ValueError):
        pc.fit(data)
