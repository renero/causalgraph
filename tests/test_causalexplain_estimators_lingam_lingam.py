import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.lingam import lingam


def test_extract_partial_orders_validates_duplicates():
    model = lingam.DirectLiNGAM(name="test")
    pk = np.array([[np.nan, 1], [1, np.nan]])
    with pytest.raises(ValueError):
        model._extract_partial_orders(pk)

    pk_empty = np.full((2, 2), np.nan)
    assert model._extract_partial_orders(pk_empty) is None


def test_search_candidate_respects_prior_knowledge():
    model = lingam.DirectLiNGAM(name="test", prior_knowledge=np.array([[np.nan, 1], [0, np.nan]]))
    model._partial_orders = np.array([[0, 1]])
    # Strong application uses partial_orders
    Uc, Vj = model._search_candidate(np.array([0, 1]))
    assert Uc == [1]
    assert Vj == []

    model_soft = lingam.DirectLiNGAM(
        name="soft", prior_knowledge=np.array([[np.nan, 0], [0, np.nan]]), apply_prior_knowledge_softly=True)
    Uc, Vj = model_soft._search_candidate(np.array([0, 1]))
    assert set(Uc) == {0, 1}
    assert Vj == []


def test_fit_and_fit_predict(monkeypatch):
    model = lingam.DirectLiNGAM(name="fit-test")

    # simplify search and adjacency estimation
    order_calls = {"calls": 0}

    def simple_search(self, X, U):
        order_calls["calls"] += 1
        return U[0]

    monkeypatch.setattr(model, "_search_causal_order", simple_search)
    monkeypatch.setattr(model, "_estimate_adjacency_matrix",
                        lambda X, prior_knowledge=None: setattr(model, "_adjacency_matrix", np.zeros((2, 2))) or model)
    monkeypatch.setattr(lingam.utils, "graph_from_adjacency",
                        lambda adjacency, node_labels, th=0.0, inverse=False, absolute_values=False: "graph")
    monkeypatch.setattr(lingam, "evaluate_graph", lambda ref_graph, dag, feature_names=None: {"ref": ref_graph})

    data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    model.fit(data)

    assert model.causal_order_ == [0, 1]
    assert model.dag == "graph"
    assert order_calls["calls"] == 2

    dag = model.fit_predict(train=data, test=None, ref_graph="ref_graph")
    assert dag == "graph"
    assert model.metrics == {"ref": "ref_graph"}
