import numpy as np
import networkx as nx
import pytest
from causalexplain.metrics.compare_graphs import (
    Metrics, _adjacency, _aupr, _binary_adj_matrix, _conf_mat,
    _conf_mat_directed, _conf_mat_undirected, _confusion_matrix, _f1,
    _is_weighted, _intersect_matrices, _negative, _positive, _precision,
    _recall, _SHD, _shallow_copy, _weighted_adjacency, evaluate_graph
)


@pytest.fixture(autouse=True)
def _isolated_matplotlib_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


@pytest.fixture
def sid_statistics():
    return {"sid": 2.0, "sidLowerBound": 1.0, "sidUpperBound": 3.0}


@pytest.fixture
def metrics_instance(sid_statistics):
    return Metrics(
        Tp=3,
        Tn=5,
        Fn=2,
        Fp=1,
        precision=0.75,
        recall=0.6,
        aupr=0.65,
        f1=0.6666666667,
        shd=2,
        sid=sid_statistics,
    )


@pytest.fixture
def directed_graph_pair():
    truth = nx.DiGraph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    est = nx.DiGraph()
    est.add_edges_from([("a", "b"), ("c", "b")])
    feature_names = ["a", "b", "c"]
    return truth, est, feature_names


@pytest.fixture
def undirected_graph_pair():
    truth = nx.Graph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    est = nx.Graph()
    est.add_edges_from([("a", "b")])
    feature_names = ["a", "b", "c"]
    return truth, est, feature_names


def test_metrics_initialization_and_to_dict(metrics_instance, sid_statistics):
    result = metrics_instance.to_dict()
    assert result["Tp"] == 3
    assert result["Tn"] == 5
    assert result["Fn"] == 2
    assert result["Fp"] == 1
    assert metrics_instance.ishd == pytest.approx(1 - (2 / 3))
    assert metrics_instance.sid_lower == sid_statistics["sidLowerBound"]
    assert metrics_instance.sid_upper == sid_statistics["sidUpperBound"]


def test_metrics_str_and_matplotlib_repr(metrics_instance):
    text = str(metrics_instance)
    assert "Precision: 0.75" in text
    assert "Recall...: 0.6" in text
    assert "[1.0..3.0]" in text
    plt_text = metrics_instance.matplotlib_repr()
    assert "> Precision: 0.75" in plt_text
    assert "> SID......: 2.0" in plt_text


def test_metrics_handles_zero_errors_in_str():
    metrics = Metrics(
        Tp=1,
        Tn=0,
        Fn=0,
        Fp=0,
        precision=1.0,
        recall=1.0,
        aupr=1.0,
        f1=1.0,
        shd=0,
        sid={"sid": 0.0, "sidLowerBound": 0.0, "sidUpperBound": 0.0},
    )
    sid_line = str(metrics).splitlines()[-1]
    assert "[" not in sid_line
    assert metrics.ishd == 0.0
    assert "> SID......: 0.0" in metrics.matplotlib_repr()


def test_shallow_copy_isolated_from_original():
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    copy = _shallow_copy(graph)
    assert isinstance(copy, nx.DiGraph)
    assert set(copy.edges()) == {("a", "b"), ("b", "c")}
    copy.add_edge("c", "a")
    assert ("c", "a") not in graph.edges()


def test_evaluate_graph_returns_none_when_truth_missing():
    predicted = nx.DiGraph()
    predicted.add_edge("a", "b")
    assert evaluate_graph(None, predicted) is None


def test_evaluate_graph_computes_metrics_for_directed_graphs(directed_graph_pair):
    truth, est, _ = directed_graph_pair
    metrics = evaluate_graph(truth, est)
    assert metrics.Tp == 1
    assert metrics.Tn == 3
    assert metrics.Fn == 1
    assert metrics.Fp == 1
    assert metrics.shd == 2
    assert metrics.sid == 3
    assert metrics.aupr == pytest.approx(5 / 9)


def test_evaluate_graph_handles_missing_nodes_and_feature_names():
    """Prediction graphs missing nodes must still produce valid metrics."""
    truth = nx.DiGraph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    truth.add_node("d")
    predicted = nx.DiGraph()
    predicted.add_edge("a", "b")
    metrics = evaluate_graph(truth, predicted,
                             feature_names=["a", "b", "c", "d"])
    assert metrics.Tp == 1
    assert metrics.Tn == 10
    assert metrics.Fn == 1
    assert metrics.Fp == 0
    assert metrics.sid == 2
    assert metrics.aupr == pytest.approx(0.78125)


def test_evaluate_graph_double_for_anticausal_toggle():
    """SHD drops when reversed edges are not double-counted."""
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    predicted = nx.DiGraph()
    predicted.add_edge("b", "a")
    default_metrics = evaluate_graph(truth, predicted)
    relaxed_metrics = evaluate_graph(
        truth, predicted, double_for_anticausal=False)
    assert default_metrics.shd == 2
    assert relaxed_metrics.shd == 1


def test_evaluate_graph_mismatched_directedness_raises():
    """Different graph types currently trigger an assertion."""
    truth = nx.Graph()
    truth.add_edge("a", "b")
    predicted = nx.DiGraph()
    predicted.add_edge("a", "b")
    with pytest.raises(AssertionError):
        evaluate_graph(truth, predicted)


def test_conf_mat_directed_wrapper(directed_graph_pair):
    truth, est, feature_names = directed_graph_pair
    assert _conf_mat(truth, est, feature_names) == (1, 3, 1, 1)


def test_conf_mat_undirected_wrapper(undirected_graph_pair):
    truth, est, feature_names = undirected_graph_pair
    assert _conf_mat(truth, est, feature_names) == (1, 1, 0, 1)


def test_confusion_matrix_directed_branch():
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b"])
    metrics = {
        "_Gm": np.array([[0, 1], [0, 0]], dtype=int),
        "_gm": np.array([[0, 1], [0, 0]], dtype=int),
    }
    assert _confusion_matrix(graph, metrics) == (1, 1, 0, 0)


def test_confusion_matrix_undirected_branch():
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b"])
    metrics = {
        "_Gm": np.array([[0, 1], [1, 0]], dtype=int),
        "_gm": np.array([[0, 0], [0, 0]], dtype=int),
    }
    assert _confusion_matrix(graph, metrics) == (0, 0, -1, 0)


def test_confusion_matrix_rejects_unknown_graph_type():
    """The helper validates that the graph is a supported NetworkX type."""
    metrics = {"_Gm": np.zeros((1, 1)), "_gm": np.zeros((1, 1))}
    with pytest.raises(TypeError):
        _confusion_matrix(object(), metrics)


def test_shd_counts_directional_disagreements_twice():
    metrics = {
        "_Gm": np.array([[0, 1], [0, 0]]),
        "_gm": np.array([[0, 0], [1, 0]]),
        "_double_for_anticausal": True,
    }
    assert _SHD(metrics) == 2
    metrics["_double_for_anticausal"] = False
    assert _SHD(metrics) == 1


def test_shd_zero_for_identical_matrices():
    metrics = {
        "_Gm": np.zeros((2, 2), dtype=int),
        "_gm": np.zeros((2, 2), dtype=int),
        "_double_for_anticausal": True,
    }
    assert _SHD(metrics) == 0


# Two identical graphs are compared.
def test_identical_graphs():
    truth = nx.DiGraph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    est = nx.DiGraph()
    est.add_edges_from([("a", "b"), ("b", "c")])
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)
    assert tp == 2
    assert tn == 4
    assert fp == 0
    assert fn == 0

# Two graphs, same nodes, with one edge different are compared.


def test_one_edge_different():
    truth = nx.DiGraph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    est = nx.DiGraph()
    est.add_edges_from([("a", "b"), ("c", "a")])
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)
    assert tp == 1
    assert tn == 3
    assert fp == 1
    assert fn == 1

# Two graphs with different edges are compared.


def test_different_edges():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("a", "c")
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 1
    assert fp == 1
    assert tn == 3
    assert fn == 1

# Two graphs with different nodes are compared.


def test_different_nodes_est_more_connections():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    truth.add_node("d")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("b", "c")
    est.add_edge("c", "d")
    feature_names = ["a", "b", "c", "d"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert tn == 9
    assert fp == 0
    assert fn == 1

# Two graphs with different nodes (truth has more connection) are compared.


def test_different_nodes_truth_more_connections():
    truth = nx.DiGraph()
    truth.add_edge("a", "b")
    truth.add_edge("b", "c")
    truth.add_edge("c", "d")
    est = nx.DiGraph()
    est.add_edge("a", "b")
    est.add_edge("b", "c")
    est.add_node("d")
    feature_names = ["a", "b", "c", "d"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 2
    assert fp == 1
    assert tn == 9
    assert fn == 0


# Two graphs with one node and no edges are compared.
def test_one_node_no_edges():
    truth = nx.DiGraph()
    truth.add_node("a")
    est = nx.DiGraph()
    est.add_node("a")
    feature_names = ["a"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 0
    assert fp == 0
    assert tn == 0
    assert fn == 0

# Two graphs with one node and a self-loop are compared.


def test_one_node_self_loop():
    truth = nx.DiGraph()
    truth.add_edge("a", "a")
    est = nx.DiGraph()
    est.add_edge("a", "a")
    feature_names = ["a"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 1
    assert fp == 0
    assert tn == 0
    assert fn == 0

# Two graphs with multiple nodes and no edges are compared.


def test_multiple_nodes_no_edges():
    truth = nx.DiGraph()
    truth.add_nodes_from(["a", "b", "c"])
    est = nx.DiGraph()
    est.add_nodes_from(["a", "b", "c"])
    feature_names = ["a", "b", "c"]

    tp, tn, fn, fp = _conf_mat_directed(truth, est, feature_names)

    assert tp == 0
    assert fp == 0
    assert tn == 6
    assert fn == 0


def test_unweighted_graph():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    assert not _is_weighted(G)


def test_weighted_graph():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1.0)
    G.add_edge("b", "c", weight=2.0)
    assert _is_weighted(G)


def test_mixed_graph():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c", weight=2.0)
    assert _is_weighted(G)


# Tests for the _binary_adj_matrix

def test_binary_adj_matrix_default_threshold():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.2)
    G.add_edge("c", "a", weight=0.0)
    expected_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    result_matrix = _binary_adj_matrix(G)
    np.testing.assert_array_equal(result_matrix, expected_matrix)


def test_binary_adj_matrix_with_threshold():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.2)
    G.add_edge("c", "a", weight=0.0)
    expected_matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    result_matrix = _binary_adj_matrix(G, threshold=0.3)
    np.testing.assert_array_equal(result_matrix, expected_matrix)


def test_binary_adj_matrix_with_absolute():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=-0.5)
    G.add_edge("b", "c", weight=0.2)
    G.add_edge("c", "a", weight=-0.4)
    expected_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    result_matrix = _binary_adj_matrix(G, absolute=True)
    np.testing.assert_array_equal(result_matrix, expected_matrix)


def test_binary_adj_matrix_with_order():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.2)
    G.add_edge("c", "a", weight=0.0)
    expected_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    result_matrix = _binary_adj_matrix(G, order=["c", "b", "a"])
    np.testing.assert_array_equal(result_matrix, expected_matrix)


# Test for the _adjacency function

def test_adjacency_unweighted_graph():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    expected_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = _adjacency(G)
    np.testing.assert_array_equal(result, expected_matrix)


def test_adjacency_weighted_graph_with_threshold():
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.8)
    G.add_edge("a", "c", weight=0.3)
    expected_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    result = _adjacency(G, threshold=0.4)
    np.testing.assert_array_equal(result, expected_matrix)


def test_adjacency_with_unoriented_order():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    order = ["b", "a", "c"]
    expected_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    result = _adjacency(G, order=order)
    np.testing.assert_array_equal(result, expected_matrix)


def test_adjacency_with_oriented_order():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    order = ["b", "a", "c"]
    expected_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    result = _adjacency(G, order=order)
    np.testing.assert_array_equal(result, expected_matrix)

# Tests for the _weighted_adjacency function


def test_weighted_adjacency_unweighted_graph():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    expected_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = _weighted_adjacency(G)
    np.testing.assert_array_equal(result, expected_matrix)


def test_weighted_adjacency_with_threshold():
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.8)
    G.add_edge("a", "c", weight=0.3)
    expected_matrix = np.array([[0, 0.5, 0], [0.5, 0, 0.8], [0, 0.8, 0]])
    result = _weighted_adjacency(G, threshold=0.4)
    np.testing.assert_array_equal(result, expected_matrix)


def test_weighted_adjacency_with_absolute():
    G = nx.Graph()
    G.add_edge("a", "b", weight=-0.5)
    G.add_edge("b", "c", weight=0.8)
    G.add_edge("a", "c", weight=-0.3)
    expected_matrix = np.array([[0, 0.5, 0], [0.5, 0, 0.8], [0, 0.8, 0]])
    result = _weighted_adjacency(G, threshold=0.4, absolute=True)
    np.testing.assert_array_equal(result, expected_matrix)


def test_weighted_adjacency_with_order():
    G = nx.Graph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.8)
    order = ["b", "a", "c"]
    expected_matrix = np.array([[0, 0.5, 0.8], [0.5, 0, 0], [0.8, 0, 0]])
    result = _weighted_adjacency(G, order=order)
    np.testing.assert_array_equal(result, expected_matrix)


def test_weighted_adjacency_oriented_with_order():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=0.5)
    G.add_edge("b", "c", weight=0.8)
    order = ["b", "a", "c"]
    expected_matrix = np.array([[0, 0, 0.8], [0.5, 0, 0], [0, 0, 0]])
    result = _weighted_adjacency(G, order=order)
    np.testing.assert_array_equal(result, expected_matrix)

# Tests for the _intersect_matrices function


def test_intersect_matrices_identical():
    mat1 = np.array([[1, 0], [0, 1]])
    mat2 = np.array([[1, 0], [0, 1]])
    expected_result = np.array([[1, 0], [0, 1]])
    result = _intersect_matrices(mat1, mat2)
    np.testing.assert_array_equal(result, expected_result)


def test_intersect_matrices_different():
    mat1 = np.array([[1, 0], [0, 1]])
    mat2 = np.array([[0, 1], [1, 0]])
    expected_result = np.array([[0, 0], [0, 0]])
    result = _intersect_matrices(mat1, mat2)
    np.testing.assert_array_equal(result, expected_result)


def test_intersect_matrices_different_shapes():
    mat1 = np.array([[1, 0, 1], [0, 1, 0]])
    mat2 = np.array([[1, 0], [0, 1]])
    try:
        _intersect_matrices(mat1, mat2)
    except ValueError as e:
        assert str(e) == "Both graphs must have the same number of nodes"

# Tests for the _positive function


def test_positive_all_positive_values():
    matrix = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[1, 2], [3, 4]])
    result = _positive(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_positive_all_negative_values():
    matrix = np.array([[-1, -2], [-3, -4]])
    expected_result = np.array([[0, 0], [0, 0]])
    result = _positive(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_positive_mixed_values():
    matrix = np.array([[1, -2], [-3, 4]])
    expected_result = np.array([[1, 0], [0, 4]])
    result = _positive(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_positive_zero_values():
    matrix = np.array([[0, 0], [0, 0]])
    expected_result = np.array([[0, 0], [0, 0]])
    result = _positive(matrix)
    np.testing.assert_array_equal(result, expected_result)

# Tests for the _negative function


def test_negative_all_positive_values():
    matrix = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[0, 0], [0, 0]])
    result = _negative(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_negative_all_negative_values():
    matrix = np.array([[-1, -2], [-3, -4]])
    expected_result = np.array([[-1, -2], [-3, -4]])
    result = _negative(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_negative_mixed_values():
    matrix = np.array([[1, -2], [-3, 4]])
    expected_result = np.array([[0, -2], [-3, 0]])
    result = _negative(matrix)
    np.testing.assert_array_equal(result, expected_result)


def test_negative_zero_values():
    matrix = np.array([[0, 0], [0, 0]])
    expected_result = np.array([[0, 0], [0, 0]])
    result = _negative(matrix)
    np.testing.assert_array_equal(result, expected_result)

# Tests for the _conf_mat_undirected function


def test_conf_mat_undirected_identical_graphs():
    truth = nx.Graph()
    truth.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    est = nx.Graph()
    est.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    feature_names = ["a", "b", "c"]
    Tp, Tn, Fp, Fn = _conf_mat_undirected(truth, est, feature_names)
    assert Tp == 3
    assert Tn == 0
    assert Fp == 0
    assert Fn == 0


def test_conf_mat_undirected_one_missing_edge():
    truth = nx.Graph()
    truth.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    est = nx.Graph()
    est.add_edges_from([("a", "b"), ("b", "c")])
    feature_names = ["a", "b", "c"]
    Tp, Tn, Fp, Fn = _conf_mat_undirected(truth, est, feature_names)
    assert Tp == 2
    assert Tn == 0
    assert Fp == 0
    assert Fn == 1


def test_conf_mat_undirected_one_extra_edge():
    truth = nx.Graph()
    truth.add_edges_from([("a", "b"), ("b", "c")])
    est = nx.Graph()
    est.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    feature_names = ["a", "b", "c"]
    Tp, Tn, Fp, Fn = _conf_mat_undirected(truth, est, feature_names)
    assert Tp == 2
    assert Tn == 0
    assert Fp == 1
    assert Fn == 0


def test_conf_mat_undirected_completely_different():
    truth = nx.Graph()
    truth.add_edges_from([("a", "b")])
    est = nx.Graph()
    est.add_edges_from([("b", "c")])
    feature_names = ["a", "b", "c"]
    Tp, Tn, Fp, Fn = _conf_mat_undirected(truth, est, feature_names)
    assert Tp == 0
    assert Tn == 1
    assert Fp == 1
    assert Fn == 1


def test_precision():
    # Test case where Tp = 0 and Fp = 0
    metrics = {'Tp': 0, 'Fp': 0}
    assert _precision(
        metrics) == 0, "Precision should be 0 when Tp and Fp are both 0"

    # Test case where Tp = 5 and Fp = 0
    metrics = {'Tp': 5, 'Fp': 0}
    assert _precision(
        metrics) == 1, "Precision should be 1 when Tp = 5 and Fp = 0"

    # Test case where Tp = 5 and Fp = 5
    metrics = {'Tp': 5, 'Fp': 5}
    assert _precision(
        metrics) == 0.5, "Precision should be 0.5 when Tp = 5 and Fp = 5"

    # Test case where Tp = 0 and Fp = 5
    metrics = {'Tp': 0, 'Fp': 5}
    assert _precision(
        metrics) == 0, "Precision should be 0 when Tp = 0 and Fp = 5"


def test_recall():
    # Test case where Tp = 0 and Fn = 0
    metrics = {'Tp': 0, 'Fn': 0}
    assert _recall(
        metrics) == 0, "Recall should be 0 when Tp and Fn are both 0"

    # Test case where Tp = 5 and Fn = 0
    metrics = {'Tp': 5, 'Fn': 0}
    assert _recall(metrics) == 1, "Recall should be 1 when Tp = 5 and Fn = 0"

    # Test case where Tp = 5 and Fn = 5
    metrics = {'Tp': 5, 'Fn': 5}
    assert _recall(
        metrics) == 0.5, "Recall should be 0.5 when Tp = 5 and Fn = 5"

    # Test case where Tp = 0 and Fn = 5
    metrics = {'Tp': 0, 'Fn': 5}
    assert _recall(metrics) == 0, "Recall should be 0 when Tp = 0 and Fn = 5"


def test_aupr():
    # Test case with perfect predictions
    metrics = {'_Gm': np.array([[1, 0], [0, 1]]),
               '_preds': np.array([[1, 0], [0, 1]])}
    assert _aupr(metrics) > 0.9, "AUPR should be high for perfect predictions"

    # Test case with random predictions
    metrics = {'_Gm': np.array([[1, 0], [0, 1]]),
               '_preds': np.array([[0.5, 0.5], [0.5, 0.5]])}
    assert 0 <= _aupr(
        metrics) <= 1, "AUPR should be between 0 and 1 for random predictions"

    # Test case with dimension mismatch
    metrics = {'_Gm': np.array([[1, 0]]), '_preds': np.array([[1, 0], [0, 1]])}
    assert _aupr(metrics) == 0.0, "AUPR should be 0 for dimension mismatch"


def test_f1():
    # Test case where precision = 0 and recall = 0
    metrics = {'precision': 0, 'recall': 0}
    assert _f1(metrics) == 0, "F1 should be 0 when precision and recall are both 0"

    # Test case where precision = 1 and recall = 1
    metrics = {'precision': 1, 'recall': 1}
    assert _f1(metrics) == 1, "F1 should be 1 when precision and recall are both 1"

    # Test case where precision = 0.5 and recall = 0.5
    metrics = {'precision': 0.5, 'recall': 0.5}
    assert _f1(
        metrics) == 0.5, "F1 should be 0.5 when precision and recall are both 0.5"
