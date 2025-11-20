import sys
import types

import numpy as np
import pandas as pd
import pytest

from causalexplain.independence import mic


class DummyProgBar:
    def start_subtask(self, *args, **kwargs):
        return types.SimpleNamespace(update_subtask=lambda *a, **k: None)


def test_pairwise_mic_raises_when_minepy_missing(monkeypatch):
    monkeypatch.setattr(mic, "ProgBar", lambda *args, **kwargs: DummyProgBar())
    monkeypatch.delitem(sys.modules, "minepy", raising=False)
    with pytest.raises(ModuleNotFoundError):
        mic.pairwise_mic(pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]}))


def test_pairwise_mic_with_stub(monkeypatch):
    data = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.5, 0.5]})
    mic_values = np.array([0.1, 0.2, 0.3])
    tic_values = np.array([0.4, 0.5, 0.6])

    monkeypatch.setattr(mic, "ProgBar", lambda *args, **kwargs: DummyProgBar())
    fake_minepy = types.SimpleNamespace(pstats=lambda *a, **k: (mic_values, tic_values))
    monkeypatch.setitem(sys.modules, "minepy", fake_minepy)

    result = mic.pairwise_mic(data, prog_bar=False)
    assert result.shape == (3, 3)
    assert np.isclose(result[0, 1], mic_values[0])
