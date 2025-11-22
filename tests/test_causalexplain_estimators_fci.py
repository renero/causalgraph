import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.fci import fci as fci_module


def test_filter_causes_only_detects_direct_relationships():
    model = fci_module.FCI(name="filter-test")
    adjacency = np.array([
        [0, -1, 2],
        [1, 0, 0],
        [1, -1, 0],
    ])

    filtered = model.filter_causes_only(adjacency)

    expected = np.zeros_like(adjacency)
    expected[0, 1] = 1  # -1 / 1 combination -> direct edge 0->1
    expected[0, 2] = 1  # 2 / 1 combination -> direct edge 0->2

    assert np.array_equal(filtered, expected)


def test_fit_predict_causes_only_filters_and_sets_attributes(monkeypatch):
    X = pd.DataFrame({"a": [0.1, 0.2], "b": [0.2, 0.3]})
    ref_graph = object()
    fci_called = {}

    class GraphStub:
        def __init__(self, matrix):
            self.graph = matrix

    def fake_fci(values, indep_test_method, alpha, depth, max_path_length,
                 verbose, show_progress, background_knowledge, node_names):
        fci_called["kwargs"] = {
            "indep_test_method": indep_test_method,
            "alpha": alpha,
            "depth": depth,
            "max_path_length": max_path_length,
            "verbose": verbose,
            "show_progress": show_progress,
            "background_knowledge": background_knowledge,
            "node_names": node_names,
        }
        matrix = np.array([[0, -1], [1, 0]])
        return GraphStub(matrix), ["edge"]

    adjacency_inputs = {}

    def fake_graph_from_adjacency(matrix, node_labels):
        adjacency_inputs["matrix"] = matrix
        adjacency_inputs["labels"] = node_labels
        return "graph_obj"

    metrics_called = {}

    def fake_evaluate_graph(ref, dag):
        metrics_called["args"] = (ref, dag)
        return {"score": 0.9}

    monkeypatch.setattr(fci_module, "fci", fake_fci)
    monkeypatch.setattr(fci_module.utils, "graph_from_adjacency",
                        fake_graph_from_adjacency)
    monkeypatch.setattr(fci_module, "evaluate_graph", fake_evaluate_graph)

    model = fci_module.FCI(
        name="experiment",
        independence_test_method="kci",
        alpha=0.01,
        depth=4,
        max_path_length=7,
        background_knowledge="bk",
        node_names=["a", "b"],
        causes_only=True,
    )

    result = model.fit_predict(X=X, ref_graph=ref_graph)

    assert result is model
    assert fci_called["kwargs"] == {
        "indep_test_method": "kci",
        "alpha": 0.01,
        "depth": 4,
        "max_path_length": 7,
        "verbose": False,
        "show_progress": False,
        "background_knowledge": "bk",
        "node_names": ["a", "b"],
    }
    assert np.array_equal(adjacency_inputs["matrix"], np.array([[0, 1], [0, 0]]))
    assert adjacency_inputs["labels"] == ["a", "b"]
    assert metrics_called["args"] == (ref_graph, "graph_obj")
    assert model.metrics == {"score": 0.9}
    assert model.dag == "graph_obj"


def test_fit_predict_without_causes_only_retains_full_adjacency(monkeypatch):
    X = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    class GraphStub:
        def __init__(self, matrix):
            self.graph = matrix

    adjacency = np.array([[0, 2], [2, 0]])

    def fake_fci(values, **_):
        return GraphStub(adjacency), ["edge"]

    adjacency_inputs = {}

    def fake_graph_from_adjacency(matrix, node_labels):
        adjacency_inputs["matrix"] = matrix
        adjacency_inputs["labels"] = node_labels
        return "full_graph"

    monkeypatch.setattr(fci_module, "fci", fake_fci)
    monkeypatch.setattr(fci_module.utils, "graph_from_adjacency",
                        fake_graph_from_adjacency)
    monkeypatch.setattr(fci_module, "evaluate_graph",
                        lambda ref_graph, dag: {"ref": ref_graph, "dag": dag})

    model = fci_module.FCI(name="no-filter", causes_only=False)
    result = model.fit_predict(X=X, ref_graph="ref")

    assert result.dag == "full_graph"
    assert np.array_equal(adjacency_inputs["matrix"], adjacency)
    assert adjacency_inputs["labels"] == ["x", "y"]
    assert model.metrics == {"ref": "ref", "dag": "full_graph"}


@pytest.mark.parametrize(
    "with_dag,metrics_available",
    [
        (True, True),
        (False, False),
    ],
)
def test_main_prints_edges_and_metrics(monkeypatch, capsys, with_dag, metrics_available):
    dataset_name = "sample"
    dataframe = pd.DataFrame({"x": [1], "y": [2]})

    monkeypatch.setattr(fci_module.pd, "read_csv", lambda path: dataframe)
    monkeypatch.setattr(fci_module.utils, "graph_from_dot_file", lambda path: "ref_graph")

    def fake_fit_predict(self, X, X_test=None, ref_graph=None):
        if with_dag:
            self.dag = nx.DiGraph()
            self.dag.add_edge("x", "y")
        else:
            class DummyPag:
                def __init__(self):
                    self._edges = [("y", "x")]

                def edges(self):
                    return list(self._edges)

            self.dag = None
            self.pag = DummyPag()
        self.metrics = {"metric": 1} if metrics_available else None
        return self

    monkeypatch.setattr(fci_module.FCI, "fit_predict", fake_fit_predict)

    fci_module.main(dataset_name, input_path="input/", output_path="output/")

    output_lines = capsys.readouterr().out.splitlines()

    if with_dag:
        assert "('x', 'y')" in output_lines
    else:
        assert "('y', 'x')" in output_lines

    if metrics_available:
        assert "{'metric': 1}" in output_lines
    else:
        assert "No metrics available" in output_lines
