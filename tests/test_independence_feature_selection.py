import numpy as np
import pytest

import causalexplain.independence.feature_selection as feature_selection
from causalexplain.independence.feature_selection import (
    find_cluster_change_point,
    select_features,
)


def test_select_features_requires_threshold_when_exhaustive():
    with pytest.raises(ValueError):
        select_features(np.array([1.0, 2.0]), ["a", "b"], exhaustive=True)


def test_select_features_exits_when_impact_below_minimum():
    values = np.array([1e-8, 2e-8, 3e-8])
    assert select_features(values, ["a", "b", "c"], min_impact=1e-6) == []


def test_select_features_handles_matrix_and_sorts_by_mean():
    values = np.array([[0.0, 0.1, 0.9], [0.0, 0.1, 0.9]])
    assert select_features(values, ["a", "b", "c"]) == ["c"]


def test_select_features_returns_shaps_when_requested():
    values = np.array([0.0, 0.0, 1.0, 2.0])
    selected, shaps = select_features(
        values,
        ["w", "x", "y", "z"],
        return_shaps=True,
        exhaustive=True,
        threshold=0.1,
    )
    assert selected == ["z", "y"]
    assert shaps == [2.0, 1.0]


def test_select_features_non_exhaustive_orders_by_cluster_change():
    values = np.array([0.1, 0.4, 0.2])
    assert select_features(values, ["a", "b", "c"]) == ["b"]


def test_select_features_exhaustive_iterates_until_threshold(monkeypatch):
    calls = []

    def fake_change_point(current_values, verbose=False):
        # Peel off one more element each time to exercise the exhaustive branch.
        calls.append(list(current_values))
        return len(current_values) - 1

    monkeypatch.setattr(feature_selection, "find_cluster_change_point", fake_change_point)
    values = np.array([0.1, 0.2, 0.3])
    result = select_features(values, ["a", "b", "c"], exhaustive=True, threshold=0.05)

    assert result == ["c", "b", "a"]
    assert calls == [[0.1, 0.2, 0.3], [0.1, 0.2], [0.1]]


def test_select_features_stops_after_max_iterations(monkeypatch):
    def non_progressing_change_point(current_values, verbose=False):
        # Returning len(current_values) means the slice never shrinks.
        return len(current_values)

    monkeypatch.setattr(
        feature_selection, "find_cluster_change_point", non_progressing_change_point
    )
    values = np.array([0.1, 0.2])
    assert select_features(values, ["a", "b"], exhaustive=True, threshold=0.05) == []


def test_select_features_emits_useful_verbose_output(capsys):
    values = np.array([0.1, 0.3, 1.0])
    select_features(values, ["a", "b", "c"], threshold=0.05, verbose=True)

    output = capsys.readouterr().out
    assert "Feature order" in output
    assert "Selected_features" in output


def test_select_features_verbose_matrix_output_lists_sums(capsys):
    values = np.array([[0.0, 0.2], [0.1, 0.3]])
    select_features(values, ["a", "b"], verbose=True)
    output = capsys.readouterr().out
    assert "Sum values" in output
    assert "(a:" in output


def test_find_cluster_change_point_handles_identical_values_and_logs(capsys):
    assert find_cluster_change_point([0, 0, 0], verbose=True) is None
    captured = capsys.readouterr()
    assert "** No clusters generated" in captured.out


def test_find_cluster_change_point_returns_none_for_short_series():
    assert find_cluster_change_point([]) is None
    assert find_cluster_change_point([1]) is None


def test_find_cluster_change_point_detects_gap_between_clusters():
    values = [0.1, 0.2, 0.8, 0.9]
    assert find_cluster_change_point(values) == 2


def test_find_cluster_change_point_reports_cluster_estimates_verbose(capsys):
    values = [0.1, 0.2, 0.8, 0.9]
    assert find_cluster_change_point(values, verbose=True) == 2
    output = capsys.readouterr().out
    assert "Est.clusters/noise" in output


def test_module_main_runs_and_logs(capsys):
    feature_selection.main()
    output = capsys.readouterr().out
    assert "Feature order" in output
    assert "threshold" in output


def test_embedded_test_function_is_intentionally_failing():
    with pytest.raises(AssertionError):
        feature_selection.test()
