import numpy as np
import networkx as nx
import pandas as pd
import pytest
import torch

from causalexplain.models import _weight_utils as wu


class DummyParam:
    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def numpy(self):
        return self._array


class DummyModule:
    def __init__(self):
        self._weights = [
            ("weight", DummyParam(np.array([[1.0, -1.0], [0.5, 0.25]]))),
            ("bias", DummyParam(np.array([0.0, 0.1]))),
        ]

    def named_parameters(self):
        return self._weights


def test_extract_weights_filters_non_weights():
    weights = wu.extract_weights(DummyModule())
    assert len(weights) == 1
    assert weights[0].shape == (2, 2)


def test_plot_helpers_no_display(monkeypatch):
    monkeypatch.setattr(wu.plt, "show", lambda *_, **__: None)
    W = np.array([[0.1, 0.2], [0.3, -0.1]])
    wu.see_weights_to_hidden(W, ["f1", "f2"], target="y")
    wu.see_weights_from_input(W, ["f1", "f2"], target="y")


def test_plot_feature_composes_from_weights(monkeypatch):
    monkeypatch.setattr(wu.plt, "show", lambda *_, **__: None)
    dummy = DummyModule()

    class Result:
        def __init__(self, model):
            self.model = model
            self.columns = ["f1", "Noise"]
            self.all_columns = ["f1", "target", "Noise"]

    res = Result(dummy)
    wu.plot_feature(res)
    wu.plot_features({"f1": res, "target": res, "Noise": res}, 3, 3, res.all_columns)


def test_layer_and_summarize_weights(monkeypatch):
    class ExtendedModule(DummyModule):
        def __init__(self):
            self._weights = [
                ("weight", DummyParam(np.array([[1.0, -1.0, 0.5], [0.5, 0.25, -0.25]]))),
                ("bias", DummyParam(np.array([0.0, 0.1, -0.1]))),
            ]

    dummy = ExtendedModule()
    monkeypatch.setattr(
        wu, "extract_weights", lambda model: list(model._weights[0][1]._array[None, ...])
    )

    class Wrapper:
        def __init__(self, model):
            self.model = model
            self.columns = ["f1", "f2", "Noise"]

    weights_map = {"f1": Wrapper(dummy), "f2": Wrapper(dummy)}
    layer_df = wu.layer_weights(weights_map["f1"], target="f1")
    assert {col for col in layer_df.columns} == {"psd_f1", "avg_f1", "med_f1"}

    summary = wu.summarize_weights(weights_map, feature_names=["f1", "f2"], scale=False)
    assert summary.shape[0] == dummy._weights[0][1]._array.shape[1]


def test_summarize_weights_scales_and_plots(monkeypatch):
    monkeypatch.setattr(wu.plt, "show", lambda *_, **__: None)
    values = np.array([[1.0, 0.5], [0.2, -0.4]])

    class Wrapper:
        def __init__(self):
            self.model = type("M", (), {"named_parameters": lambda *_: [("weight", DummyParam(values))]})
            self.columns = ["a", "Noise"]

    weights_map = {"a": Wrapper(), "b": Wrapper()}
    scaled = wu.summarize_weights(weights_map, feature_names=["a", "b"], scale=True)
    assert np.allclose(scaled.mean().values, 0, atol=1e-6)

    weights_df = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]],
        columns=["psd_a", "med_a", "avg_a", "psd_b", "med_b", "avg_b"],
        index=["a", "b"],
    )
    relationships = wu.identify_relationships(
        weights_df, feature_names=["a", "b"], plot=True, eps=0.5, min_counts=1
    )
    assert set(relationships.keys()) == {"a", "b"}


def test_identify_relationships_detects_sparse_clusters():
    columns = [
        "psd_a",
        "med_a",
        "avg_a",
        "psd_b",
        "med_b",
        "avg_b",
        "psd_c",
        "med_c",
        "avg_c",
    ]
    values = [
        [0.1, 0.1, 0.2, 0.5, 0.5, 0.6, 1.0, 1.1, 1.2],
        [2.0, 1.5, 1.0, 2.5, 2.0, 2.1, 2.2, 2.3, 2.4],
        [4.0, 3.5, 3.0, 3.5, 3.0, 3.1, 3.2, 3.3, 3.4],
    ]
    data = pd.DataFrame(values, columns=columns, index=["a", "b", "c"])

    rels = wu.identify_relationships(data, feature_names=["a", "b", "c"], plot=False, eps=0.1, min_counts=1)
    assert set(rels.keys()) == {"a", "b", "c"}


def test_shap_value_helpers(monkeypatch):
    class FakeDeep:
        def __init__(self, model, data):
            self.model = model
            self.data = data

        def shap_values(self, values):
            return np.array(values)

    class DummyLoader:
        def __init__(self, features):
            self.dataset = type("DS", (), {"features": features})

    class DummyModel:
        def __init__(self):
            self.model = object()
            self.train_loader = DummyLoader(np.array([[1.0, 2.0], [3.0, 4.0]]))

    monkeypatch.setattr(wu.shap, "DeepExplainer", FakeDeep)

    dummy_model = DummyModel()
    shap_values = wu._get_shap_values(dummy_model)
    assert shap_values.shape == dummy_model.train_loader.dataset.features.shape

    avg = wu._average_shap_values({"col": shap_values}, ["col"], abs=True)
    assert avg.shape[0] == 1
    signed_avg = wu._average_shap_values({"col": shap_values}, ["col"], abs=False)
    assert signed_avg.shape == avg.shape


def test_find_shap_elbow_uses_knee_locator(monkeypatch):
    class FakeKnee:
        def __init__(self, *_, **__):
            self.elbow = 1

        def plot_knee(self, *_, **__):
            return None

    monkeypatch.setattr(wu.kneed, "KneeLocator", FakeKnee)
    values = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = wu._find_shap_elbow(values, plot=True, verbose=True)
    assert threshold >= 0


def test_identify_edges_and_orientation():
    avg_shaps = np.array([[0.9], [0.8]])
    threshold = {"a": 0.5, "b": 0.5}
    edges = wu._identify_edges(avg_shaps, ["a", "b"], threshold)
    assert ("b", pytest.approx(avg_shaps[0][0])) in edges["a"]

    g = nx.DiGraph()
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "a", weight=0.8)
    oriented = wu._orient_edges_based_on_shap(g, verbose=True)
    assert oriented.has_edge("a", "b") != oriented.has_edge("b", "a")


def test_remove_asymmetric_shap_edges():
    relations = {"a": [("b", 1.0)], "b": []}
    cleaned = wu._remove_asymmetric_shap_edges(relations)
    assert cleaned["a"] == []
    assert cleaned["b"] == []


def test_infer_causal_relationships(monkeypatch):
    class FakeProgBar:
        def start_subtask(self, *args, **kwargs):
            return self

        def update_subtask(self, *_, **__):
            return None

        def remove(self, *_, **__):
            return None

    class FakeKnee:
        def __init__(self, *_, **__):
            self.elbow = 1

        def plot_knee(self, *_, **__):
            return None

    class FakeDeep:
        def __init__(self, model, data):
            self.model = model
            self.data = data

        def shap_values(self, values):
            return np.array(values)

    class DummyLoader:
        def __init__(self, features):
            self.dataset = type("DS", (), {"features": features})

    class DummyModel:
        def __init__(self, name):
            self.model = object()
            data = np.array([[1.0, 2.0], [2.0, 1.0]])
            self.train_loader = DummyLoader(data)
            self.name = name

    monkeypatch.setattr(wu.shap, "DeepExplainer", FakeDeep)
    monkeypatch.setattr(wu.kneed, "KneeLocator", FakeKnee)
    monkeypatch.setattr(wu, "ProgBar", FakeProgBar)
    monkeypatch.setattr(
        wu,
        "graph_from_dictionary",
        lambda d: (
            lambda g: (
                [g.add_edge(u, v, weight=w) for u, rels in d.items() for v, w in rels],
                g,
            )[1]
        )(nx.DiGraph()),
    )

    models = {name: DummyModel(name) for name in ["a", "b"]}
    result = wu.infer_causal_relationships(
        trained_models=models,
        feature_names=list(models.keys()),
        prune=True,
        plot=False,
        prog_bar=True,
    )

    assert "graph" in result
    assert isinstance(result["graph"], nx.DiGraph)
