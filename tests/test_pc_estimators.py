import pandas as pd
import pytest

from causalexplain.estimators.pc import estimators


def test_convert_args_tuple_wraps_parents():
    calls = {}

    @estimators.convert_args_tuple
    def dummy(_self, variable, parents, complete_samples_only, weighted):
        calls["parents"] = parents
        return parents

    result = dummy(object(), "v", ["a", "b"], None, False)
    assert isinstance(result, tuple)
    assert calls["parents"] == ("a", "b")


def test_init_data_collects_state_names_and_validates():
    data = pd.DataFrame({"A": ["x", "x", "y"], "B": ["a", "b", "a"]})
    est = estimators.StructureEstimator()
    est._init_data(data=data, state_names=None, complete_samples_only=False)
    assert est.state_names["A"] == ["x", "y"]

    with pytest.raises(ValueError):
        est._init_data(data=data, state_names={"A": ["x"], "B": ["a", "b"]})


def test_state_counts_with_and_without_parents():
    data = pd.DataFrame({"A": ["x", "x", "y", "y"], "B": ["a", "b", "a", "b"]})
    est = estimators.StructureEstimator()
    est._init_data(data=data, complete_samples_only=True)

    counts = est.state_counts("A")
    assert counts.loc["x", "A"] == 2
    assert counts.loc["y", "A"] == 2

    counts_with_parent = est.state_counts("B", parents=["A"])
    assert counts_with_parent.loc["a", ("x",)] == 1
    assert counts_with_parent.loc["b", ("y",)] == 1


def test_state_counts_handles_missing_and_weighted_flag():
    data = pd.DataFrame({"A": ["x", None], "B": ["a", "b"]})
    est = estimators.StructureEstimator()
    est._init_data(data=data, complete_samples_only=False)

    counts = est.state_counts("B", parents=["A"], complete_samples_only=False)
    assert counts.loc["a", ("x",)] == 1

    with pytest.raises(ValueError):
        est.state_counts("A", weighted=True)
