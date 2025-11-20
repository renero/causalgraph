import networkx as nx
import pytest

from causalexplain.independence.cond_indep import (
    ConditionalIndependencies,
    SufficientSets,
    get_backdoor_paths,
    get_conditional_independencies,
    get_paths,
    get_sufficient_sets,
    get_sufficient_sets_for_pair,
    find_colliders_in_path,
)


def test_conditional_independencies_formats_and_deduplicates():
    ci = ConditionalIndependencies()
    assert repr(ci) == "{}"

    ci.add("A", "B")
    ci.add("B", "C", ["D"])
    ci.add("B", "C", ["D"])  # duplicate

    text = str(ci)
    assert "A ⊥ B" in text
    assert "B ⊥ C | ('D',)" in text
    with pytest.raises(ValueError):
        repr(ci)


def test_sufficient_sets_add_and_repr_and_str():
    ss = SufficientSets()
    assert str(ss) == "No sufficient sets found"
    assert repr(ss) == "[]"

    ss.add([("X", "Y"), ("A", "B")])
    ss.add([("X", "Y")])  # already present
    rendered = str(ss)
    assert "('X', 'Y')" in rendered and "('A', 'B')" in rendered
    assert repr(ss).startswith("[('X', 'Y'), ('A', 'B')]")


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ("x", "y", []),  # nodes missing
        ("x", "x", []),  # identical nodes
    ],
)
def test_get_backdoor_paths_handles_missing_and_same_nodes(x, y, expected):
    dag = nx.DiGraph()
    dag.add_edges_from([("a", "b"), ("b", "c")])
    assert get_backdoor_paths(dag, x, y) == expected


def test_get_backdoor_paths_returns_only_paths_starting_towards_source():
    dag = nx.DiGraph()
    dag.add_edges_from([("m", "x"), ("m", "z"), ("z", "y"), ("k", "x")])
    # Only paths where the first edge aims into x qualify
    paths = get_backdoor_paths(dag, "x", "y")
    assert paths == [["x", "m", "z", "y"]]


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ("x", "y", []),
        ("x", "x", []),
        ("x", "z", [["x", "z"]]),
    ],
)
def test_get_paths_branches(x, y, expected):
    dag = nx.DiGraph()
    dag.add_edges_from([("x", "z")])
    assert get_paths(dag, x, y) == expected


def test_find_colliders_in_path_varied_cases():
    dag = nx.DiGraph()
    dag.add_edges_from([("a", "b"), ("c", "b"), ("b", "d")])
    assert find_colliders_in_path(dag, ["a", "b", "d"]) == set()
    assert find_colliders_in_path(dag, ["a", "b", "c"]) == {"b"}
    assert find_colliders_in_path(dag, ["a", "b"]) == set()


def test_get_sufficient_sets_for_pair_filters_descendants_and_colliders():
    dag = nx.DiGraph()
    dag.add_edges_from([("z", "x"), ("z", "y"), ("x", "z2"), ("z2", "y")])
    assert get_sufficient_sets_for_pair(dag, "x", "y") == [["z"]]

    # Introduce collider so sets are discarded
    dag.add_edge("y", "z")
    assert get_sufficient_sets_for_pair(dag, "x", "y") == []


def test_get_sufficient_sets_collects_all_pairs():
    dag = nx.DiGraph()
    dag.add_edges_from([("z", "x"), ("z", "y")])
    suff_sets = get_sufficient_sets(dag)
    assert ["z"] in suff_sets._cache


def test_get_conditional_independencies_tracks_paths_and_colliders():
    dag = nx.DiGraph()
    dag.add_edges_from([("x", "z"), ("z", "y")])
    cond_indeps = get_conditional_independencies(dag, verbose=False)

    # x and y are connected through z with no colliders -> conditioning on z
    assert ("x", "y", ("z",)) in cond_indeps._cache

    # Nodes without open paths become unconditioned independencies
    isolated = nx.DiGraph()
    isolated.add_nodes_from(["a", "b"])
    empty_cond = get_conditional_independencies(isolated, verbose=False)
    assert ("a", "b", None) in empty_cond._cache
