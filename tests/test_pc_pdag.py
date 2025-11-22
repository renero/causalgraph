import networkx as nx

from causalexplain.estimators.pc.pdag import PDAG


def test_copy_and_to_dag_orientation():
    pdag = PDAG(directed_ebunch=[("X", "Y")], undirected_ebunch=[("Y", "Z")])

    copied = pdag.copy()
    assert copied.directed_edges == pdag.directed_edges
    assert copied.undirected_edges == pdag.undirected_edges

    dag = pdag.to_dag()
    assert isinstance(dag, nx.DiGraph)
    assert dag.has_edge("X", "Y")
    assert (dag.has_edge("Y", "Z") or dag.has_edge("Z", "Y"))
