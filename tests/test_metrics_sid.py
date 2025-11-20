import importlib
import pathlib
import sys
from typing import Iterable

import networkx as nx
import numpy as np
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

sid_module = importlib.import_module("causalexplain.metrics.SID")


@pytest.fixture(autouse=True)
def _matplotlib_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _networkx_matrix_aliases(monkeypatch):
    """NetworkX>=3 removed matrix helpers; reintroduce them so SID utilities work unchanged."""
    if not hasattr(nx, "from_numpy_matrix"):
        monkeypatch.setattr(
            nx, "from_numpy_matrix", nx.from_numpy_array, raising=False
        )
    if not hasattr(nx, "to_numpy_matrix"):
        monkeypatch.setattr(
            nx,
            "to_numpy_matrix",
            lambda graph, **kwargs: np.matrix(nx.to_numpy_array(graph, **kwargs)),
            raising=False,
        )


@pytest.fixture
def restore_debug_flag():
    original = sid_module.DEBUG
    yield
    sid_module.DEBUG = original


def test_debug_prints_message_when_flag_enabled(capsys, restore_debug_flag):
    sid_module.DEBUG = True
    sid_module.debug_("hello", "world")
    captured = capsys.readouterr().out
    assert "hello world" in captured


def test_pm_outputs_matrix_rows(capsys, restore_debug_flag):
    sid_module.DEBUG = True
    sid_module.pm_(np.array([[1, 0], [0, 2]]))
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("00")
    assert output[-1].strip().endswith("2")


def test_all_dags_intern_returns_column_major_flatten():
    gm = np.matrix([[1, 2], [3, 4]])
    a = np.zeros((2, 2))
    result = sid_module.allDagsIntern(gm, a, np.array([0, 1]))
    assert np.array_equal(result, np.array([[1, 3, 2, 4]]))


def test_all_dags_intern_raises_on_partially_directed_input():
    gm = np.identity(2)
    a = np.array([[0, 1], [0, 0]])
    with pytest.raises(ValueError):
        sid_module.allDagsIntern(gm, a, np.array([0, 1]))


def test_all_dags_jonas_returns_negative_one_for_directed_component():
    adj = np.matrix([[0, 1], [0, 0]])
    assert sid_module.allDagsJonas(adj, np.array([0, 1])) == -1


def test_all_dags_jonas_defers_to_all_dags_intern_for_trivial_component():
    adj = np.matrix([[0, 0], [0, 0]])
    result = sid_module.allDagsJonas(adj, np.array([0]))
    assert np.array_equal(result, np.array([[0, 0, 0, 0]]))


@pytest.mark.parametrize("sparse", [False, True])
def test_compute_path_matrix_matches_transitive_closure(sparse):
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    matrix = sid_module.computePathMatrix(adjacency, spars=sparse)
    matrix = matrix.toarray() if sparse else matrix
    expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
    assert np.array_equal(matrix, expected)


def test_compute_path_matrix_two_blocks_with_conditioning():
    """Conditioning on a node should zero-out downstream reachability in PathMatrix2."""
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int)
    base = sid_module.computePathMatrix(adjacency)
    conditioned = sid_module.computePathMatrix2(adjacency.copy(), [1], base)
    assert not bool(conditioned[0, 2])
    assert base[0, 2]


def test_compute_caus_order_prefers_sources_first():
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    assert sid_module.compute_caus_order(adjacency) == [1, 2, 3]


def test_dag2cpdag_adj_identity_for_chain_graph(monkeypatch):
    """dag2cpdagAdj should preserve a fully directed chain."""
    monkeypatch.setattr(
        sid_module,
        "compute_caus_order",
        lambda adj: list(range(adj.shape[0])),
    )
    monkeypatch.setattr(sid_module, "dag2cpdag", lambda graph: graph)
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    cpdag = sid_module.dag2cpdagAdj(adjacency)
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.array_equal(cpdag, expected)


def test_dag2cpdag_adj_returns_original_when_empty():
    adjacency = np.zeros((2, 2))
    assert sid_module.dag2cpdagAdj(adjacency) is adjacency


def test_dsep_adji_reports_reachability_from_source():
    """dSepAdji should mark descendants of a node reachable when nothing is conditioned."""
    adjacency = np.matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    cond_set: Iterable[int] = []
    path = sid_module.computePathMatrix(adjacency.copy())
    pruned = sid_module.computePathMatrix2(adjacency.copy(), cond_set, path)
    result = sid_module.dSepAdji(
        adjacency.copy(),
        i=0,
        condSet=np.array(cond_set, dtype=int),
        PathMatrix=path,
        PathMatrix2=pruned,
        spars=False,
        p=adjacency.shape[1],
    )
    assert np.array_equal(result["reachableJ"], np.array([True, True, False]))
    assert not result["reachableOnNonCausalPath"].any()


def test_unique_rows_returns_first_occurrence_indices():
    matrix = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    assert sid_module.unique_rows(matrix) == [0, 2]


def test_get_ind_in_all_others_identifies_matching_parent_sets():
    mmm = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]])
    unique = np.array([0])
    parents = np.array([0, 2])
    matches = sid_module.get_indInAllOthers(2, mmm, unique, parents, 1, np.array([1, 2]))
    assert np.array_equal(matches, np.array([0]))


def test_get_ind_in_all_others_returns_empty_when_no_match():
    mmm = np.array([[1, 0, 1], [0, 1, 0]])
    result = sid_module.get_indInAllOthers(
        2,
        mmm,
        np.array([0]),
        np.array([0, 2]),
        1,
        np.array([1]),
    )
    assert result.size == 0


def test_hamming_distance_switches_formulas():
    g1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    g2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    assert sid_module.hammingDist(g1, g1) == 0
    assert sid_module.hammingDist(g1, g2, allMistakesOne=True) == 1
    assert sid_module.hammingDist(g1, g2, allMistakesOne=False) == 2


def test_sid_sparse_branch_surfaces_sparse_limitations():
    """When requesting sparse paths the current implementation raises during np.isnan checks."""
    graph = np.array([[0, 1], [0, 0]])
    with pytest.raises(TypeError):
        sid_module.SID(graph, graph, spars=True)


# Legacy SID consistency tests (formerly in test_metrics_SID_legacy.py)


def _is_dag(adj_matrix: np.ndarray) -> bool:
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(graph)


def _compared_sid(trueGraph: np.ndarray, estGraph: np.ndarray, sid_fn) -> dict:
    if not _is_dag(estGraph):
        return {"sid": 0, "sidLowerBound": 0.0, "sidUpperBound": 0.0}
    if not isinstance(trueGraph, np.ndarray):
        trueGraph = np.array(trueGraph)
    if not isinstance(estGraph, np.ndarray):
        estGraph = np.array(estGraph)
    trueGraph = np.matrix(trueGraph)
    estGraph = np.matrix(estGraph)
    s = sid_fn(trueGraph, estGraph, edge_direction="from row to column")
    return {"sid": s[1], "sidLowerBound": s[0], "sidUpperBound": s[0]}


def _random_dag(size: int, probability: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    adj = rng.binomial(1, probability, size=(size, size)).astype(np.int8)
    adj = np.triu(adj, 1)
    perm = rng.permutation(size)
    return adj[perm, :][:, perm]


@pytest.fixture
def gadjid_sid():
    mod = pytest.importorskip("gadjid", reason="gadjid package required for legacy SID tests")
    return mod.sid


def test_manual_sid_matches_expected(gadjid_sid):
    G = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    H1 = np.array(
        [
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    H2 = np.array(
        [
            [0, 0, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    H3 = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )
    H4 = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
        ]
    )
    H5 = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    sid1 = sid_module.SID(G, H1)
    assert sid1["sid"] == 0
    sid2 = sid_module.SID(G, H2)
    assert sid2["sid"] == 8
    sid3 = sid_module.SID(G, H3)
    assert sid3["sidLowerBound"] == 0
    assert sid3["sidUpperBound"] == 15
    sid4 = sid_module.SID(G, H4)
    assert sid4["sidLowerBound"] == 8
    assert sid4["sidUpperBound"] == 16
    sid5 = sid_module.SID(G, H5)
    assert sid5["sid"] == 12


def test_compared_sid_consistent_with_sid(gadjid_sid):
    n = 20
    threshold = 0.2
    count = 0
    for _ in range(n):
        p = np.random.randint(3, 20)
        G = _random_dag(p, 0.2)
        H = G.copy()
        indices = np.where(G == 1)
        for i in range(len(indices[0])):
            if np.random.random() > threshold:
                H[indices[0][i]][indices[1][i]] = 0
        sid1 = sid_module.SID(G, H, output=False)
        sid2 = _compared_sid(G, H, sid_fn=gadjid_sid)
        if sid1["sid"] == sid2["sid"]:
            count += 1
    assert count / n == 1.0
