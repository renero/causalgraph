import numpy as np
import pytest

from causalexplain.independence.feature_selection import (
    find_cluster_change_point,
    select_features,
)


def test_select_features_requires_threshold_when_exhaustive():
    with pytest.raises(ValueError):
        select_features(np.array([1.0, 2.0]), ["a", "b"], exhaustive=True)


def test_select_features_with_return_shaps_and_exhaustive():
    values = np.array([0.0, 0.0, 1.0, 2.0])
    features = ["w", "x", "y", "z"]
    selected, shaps = select_features(
        values, features, return_shaps=True, exhaustive=True, threshold=0.1
    )
    assert selected == ["z", "y"]
    assert shaps == [2.0, 1.0]


def test_select_features_handles_matrix_input_and_min_impact():
    values = np.array([[0.0, 0.0], [0.3, 0.6]])
    features = ["a", "b"]
    result = select_features(values, features, min_impact=0.2)
    assert result == ["b", "a"]


@pytest.mark.parametrize(
    "series,expected",
    [
        ([1], None),
        ([], None),
        ([1, 5, 10, 50], 3),
    ],
)
def test_find_cluster_change_point_internal(series, expected):
    result = find_cluster_change_point(series)
    assert result == expected
