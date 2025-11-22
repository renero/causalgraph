import pytest

from causalexplain.estimators.cam import train_gp


def test_train_gp_not_implemented():
    with pytest.raises(NotImplementedError):
        train_gp.train_gp([[0.0]], [0.0])
