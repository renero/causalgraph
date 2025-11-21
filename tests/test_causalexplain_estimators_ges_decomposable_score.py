import pytest

from causalexplain.estimators.ges import decomposable_score


def test_local_score_caches_results(monkeypatch):
    calls = {"compute": 0}

    class Dummy(decomposable_score.DecomposableScore):
        def _compute_local_score(self, x, pa):
            calls["compute"] += 1
            return x + len(pa)

    score = Dummy(data=None, cache=True, debug=0)

    first = score.local_score(1, {2, 3})
    second = score.local_score(1, {3, 2})  # same set different order hits cache
    assert first == 3
    assert second == 3
    assert calls["compute"] == 1


def test_local_score_without_cache_calls_each_time():
    class Dummy(decomposable_score.DecomposableScore):
        def _compute_local_score(self, x, pa):
            return x + len(pa)

    score = Dummy(data=None, cache=False)
    assert score.local_score(0, set()) == 0
    assert score.local_score(0, set()) == 0
