import numpy as np
import pandas as pd
import pytest

from causalexplain.estimators.pc import ci_tests


def test_power_divergence_invalid_z():
    df = pd.DataFrame({"X": [0, 1], "Y": [1, 0]})
    with pytest.raises(Exception):
        ci_tests.power_divergence("X", "Y", Z=None, data=df)


def test_power_divergence_boolean_and_tuple():
    df = pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 0, 1, 1]})
    chi, p_value, dof = ci_tests.power_divergence("X", "Y", Z=[], data=df, boolean=False)
    assert chi >= 0
    assert 0 <= p_value <= 1
    assert dof == 1

    independent = ci_tests.power_divergence("X", "Y", Z=[], data=df, boolean=True, significance_level=0.0)
    assert isinstance(independent, (bool, np.bool_))


def test_power_divergence_with_conditionals():
    df = pd.DataFrame(
        {
            "X": [0, 0, 1, 1, 0, 1],
            "Y": [0, 1, 0, 1, 0, 1],
            "Z": [0, 0, 0, 0, 1, 1],
        }
    )
    result = ci_tests.power_divergence("X", "Y", Z=["Z"], data=df, boolean=True, significance_level=0.5)
    assert isinstance(result, (bool, np.bool_))


def test_pearsonr_validation_errors():
    df = pd.DataFrame({"X": [0], "Y": [1]})
    with pytest.raises(ValueError):
        ci_tests.pearsonr("X", "Y", Z=None, data=df)
    with pytest.raises(ValueError):
        ci_tests.pearsonr("X", "Y", Z=[], data=[[1, 2]])


def test_pearsonr_boolean_mode_and_residuals():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"X": rng.randn(50), "Y": rng.randn(50) * 0.5 + 1, "Z": rng.randn(50)})

    result_bool = ci_tests.pearsonr("X", "Y", Z=[], data=df, boolean=True, significance_level=0.01)
    assert isinstance(result_bool, bool)

    coef, p_val = ci_tests.pearsonr("X", "Y", Z=["Z"], data=df, boolean=False)
    assert abs(coef) < 0.3
    assert 0 <= p_val <= 1
