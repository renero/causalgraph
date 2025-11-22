import pytest

from causalexplain.common import metrics_config


def test_global_metric_types_are_lists_of_strings():
    # Ensure both metric collections are exposed and non-empty.
    assert isinstance(metrics_config.global_metric_types, list)
    assert isinstance(metrics_config.global_nc_metric_types, list)
    assert all(isinstance(m, str) for m in metrics_config.global_metric_types)
    assert all(isinstance(m, str) for m in metrics_config.global_nc_metric_types)
    # Spot-check a couple of known entries to guard against regressions.
    assert "f1" in metrics_config.global_metric_types
    assert "r2" in metrics_config.global_nc_metric_types
