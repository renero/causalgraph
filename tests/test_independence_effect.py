import networkx as nx
import pandas as pd
import pytest

from causalexplain.independence import effect


class FakeEstimate:
    def __init__(self, value):
        self.value = value


class FakeRefuter:
    def __init__(self, p_value):
        self.refutation_result = {"p_value": p_value}


class FakeCausalModel:
    def __init__(self, *, data, treatment, outcome, graph):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.graph = graph
        self.calls = []

    def identify_effect(self, proceed_when_unidentifiable=True):
        self.calls.append(("identify", proceed_when_unidentifiable))
        return "identified"

    def estimate_effect(self, estimand, target_units="ate", method_name=None):
        self.calls.append(("estimate", estimand, target_units, method_name))
        return FakeEstimate(1.5)

    def refute_estimate(self, estimand, estimate, method_name=None):
        self.calls.append(("refute", estimand, estimate, method_name))
        return FakeRefuter(0.25)


def test_estimate_edge_uses_causal_model(monkeypatch):
    data = pd.DataFrame({"t": [0, 1, 0], "o": [1, 2, 3]})
    model_calls = {}

    def fake_model(**kwargs):
        cm = FakeCausalModel(**kwargs)
        model_calls["instance"] = cm
        return cm

    monkeypatch.setattr(effect, "CausalModel", fake_model)

    ate, pval = effect.estimate_edge(nx.DiGraph(), "t", "o", data, verbose=True)
    assert ate == 1.5 and pval == 0.25

    cm = model_calls["instance"]
    assert cm.calls[0][0] == "identify"
    assert cm.calls[1][0] == "estimate"
    assert cm.calls[2][0] == "refute"


def test_estimate_sets_attributes_and_respects_copy(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    data = pd.DataFrame({"a": [0, 1], "b": [1, 2]})

    monkeypatch.setattr(effect, "estimate_edge", lambda *args, **kwargs: ("ate", 0.1))

    modified = effect.estimate(graph, data, in_place=False)
    assert graph.get_edge_data("a", "b") == {}
    assert modified.get_edge_data("a", "b")["ate"] == "ate"
    assert modified.get_edge_data("a", "b")["refute_pval"] == 0.1
