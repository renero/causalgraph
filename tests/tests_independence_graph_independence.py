import networkx as nx
import numpy as np
import pandas as pd
import pytest

import causalexplain.independence.graph_independence as gi_mod
from causalexplain.independence.graph_independence import GraphIndependence


class StubHsic:
    def __init__(self, pvalue):
        self.pvalue = pvalue
        self.calls = []

    def test(self, x, y):
        self.calls.append((x, y))
        return 0.0, self.pvalue


def _base_instance(condsize=1):
    base_graph = nx.DiGraph([("x", "y"), ("x", "z")])
    gi = GraphIndependence(
        base_graph, condlen=1, condsize=condsize, prog_bar=False, verbose=True
    )
    gi.G_skl = base_graph.copy()
    gi.data = pd.DataFrame({"x": [0, 1], "y": [1, 2], "z": [2, 3]})
    gi.sepset = {("x", "y"): (), ("y", "x"): ()}
    return gi


def test_gen_cond_sets_and_empty_helper():
    gi = _base_instance(condsize=0)
    assert gi._gen_cond_sets("x", "y", 0) == [()]

    gi.G_skl = nx.DiGraph([("x", "y"), ("x", "z"), ("y", "z")])
    conds = gi._gen_cond_sets("x", "y", 1)
    assert conds == [("z",)]
    assert gi._empty([]) and not gi._empty([("z",)])


def test_cond_indep_test_branches(monkeypatch):
    gi = _base_instance()

    # No conditioning set
    monkeypatch.setattr(gi_mod, "Hsic", lambda: StubHsic(0.51))
    indep, _, pval = gi._cond_indep_test(
        gi.data["x"].values.reshape(-1, 1),
        gi.data["y"].values.reshape(-1, 1),
        None,
    )
    assert indep and pval == 0.51

    # With conditioning set goes through residuals branch
    monkeypatch.setattr(gi_mod, "Hsic", lambda: StubHsic(0.01))
    monkeypatch.setattr(gi, "_residuals", lambda *args: (np.array([1]), np.array([2])))
    indep, _, pval = gi._cond_indep_test(
        gi.data["x"].values.reshape(-1, 1),
        gi.data["y"].values.reshape(-1, 1),
        gi.data[["z"]].values,
    )
    assert not indep and pval == 0.01


def test_test_cond_independence_updates_graph(monkeypatch):
    gi = _base_instance()
    gi.G_skl.add_edge("y", "x")

    monkeypatch.setattr(
        gi, "_cond_indep_test", lambda *args, **kwargs: (True, 0.0, 0.2)
    )
    result = gi._test_cond_independence("x", "y", [("z",), ()])

    assert result == ("x", "y", ("z",))
    assert not gi.G_skl.has_edge("x", "y")
    assert gi.sepset[("x", "y")] == ("z",)


def test_remove_independent_edges_records_actions(monkeypatch):
    gi = _base_instance()
    gi.actions = None
    monkeypatch.setattr(gi, "_gen_cond_sets", lambda *_, **__: [("z",)])
    monkeypatch.setattr(
        gi,
        "_test_cond_independence",
        lambda x, y, cond: ("x", y, ("z",)) if y == "y" else None,
    )

    gi._remove_independent_edges("x", condlen=0, condsize=1)
    assert gi.actions["x"] == [("y", ("z",))]


def test_compute_cond_indep_pvals(monkeypatch):
    gi = _base_instance()
    gi.feature_names = list(gi.data.columns)
    monkeypatch.setattr(gi, "_cond_indep_test", lambda *args, **kwargs: (False, 0.0, 0.3))

    pvals = gi.compute_cond_indep_pvals()
    assert len(pvals) == 6  # 3 features -> 3*2 ordered pairs
    assert all(val == 0.3 for val in pvals.values())
