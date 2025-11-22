import pandas as pd
import pytest

from causalexplain.independence import edge_orientation


class DummyResult:
    def __init__(self, pvalue):
        self.pvalue = pvalue


class DummyHsic:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = []

    def test(self, x, y, reps=None):
        self.calls.append((x, y, reps))
        return DummyResult(self.sequence.pop(0))


def _patch_dependencies(monkeypatch, pvalues):
    dummy = DummyHsic(pvalues)
    monkeypatch.setattr(edge_orientation, "Hsic", lambda: dummy)
    monkeypatch.setattr(
        edge_orientation,
        "fit_and_get_residuals",
        lambda *args, **kwargs: args[1],  # pass through second array
    )
    return dummy


@pytest.mark.parametrize(
    "pvalues,expected",
    [
        ([0.2, 0.05], 1),
        ([0.01, 0.3], -1),
        ([0.1, 0.1], 0),
    ],
)
def test_get_edge_orientation_branches(monkeypatch, pvalues, expected):
    data = pd.DataFrame({"x": [0, 1, 2], "y": [2, 1, 0]})
    dummy = _patch_dependencies(monkeypatch, pvalues)

    result = edge_orientation.get_edge_orientation(
        data, "x", "y", iters=5, method="gpr", verbose=True
    )

    assert result == expected
    assert dummy.calls[0][2] == 5  # reps forwarded


def test_get_edge_orientation_invalid_method_propagates(monkeypatch):
    data = pd.DataFrame({"x": [0, 1], "y": [1, 0]})
    with pytest.raises(ValueError):
        edge_orientation.get_edge_orientation(data, "x", "y", method="bad")
