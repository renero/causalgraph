import pytest

from causalexplain.estimators.pc.independencies import IndependenceAssertion, Independencies


def test_independence_assertion_validation_and_str():
    with pytest.raises(ValueError):
        IndependenceAssertion(event2="Y")

    assertion = IndependenceAssertion("A", ["B", "C"], "D")
    assert "A" in assertion.__str__()
    assert assertion == IndependenceAssertion(["B", "C"], "A", "D")


def test_independencies_add_and_contains():
    ind = Independencies(["X", "Y"], ["A", "B", ["C"]])
    assert len(ind.get_assertions()) == 2
    assert ind == Independencies(["Y", "X"], ["B", "A", ["C"]])
    assert ind.get_all_variables() == frozenset({"X", "Y", "A", "B", "C"})

    with pytest.raises(TypeError):
        "invalid" in ind


def test_independencies_closure_contains_expected():
    ind = Independencies(["A", ["B", "C"]])
    closure = ind.closure()
    assert closure != ind
    assert any("A" in str(item) for item in closure.get_assertions())


def test_latex_string_and_repr():
    assertion = IndependenceAssertion("U", "V")
    assert "\\perp" in assertion.latex_string()
    ind = Independencies(assertion)
    latex = ind.latex_string()
    assert isinstance(latex, list)
