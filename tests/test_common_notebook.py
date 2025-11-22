import types

import pandas as pd
import pytest

from causalexplain.common import notebook


class DummyEstimator:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.fit_args = None
        self.predict_args = None

    def fit(self, *args, **kwargs):
        self.fit_args = (args, kwargs)
        return self

    def predict(self, *args, **kwargs):
        self.predict_args = (args, kwargs)
        return self

    def fit_predict(self, *args, **kwargs):
        self.fit_args = (args, kwargs)
        return self


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "data.csv"
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df.to_csv(path, index=False)
    return path


def test_check_model_type_rejects_unknown():
    exp = notebook.Experiment.__new__(notebook.Experiment)
    # Initialize BaseExperiment portion without touching filesystem
    notebook.BaseExperiment.__init__(
        exp,
        input_path="",
        output_path="",
        train_anyway=False,
        save_anyway=False,
        scale=False,
        train_size=0.9,
        random_state=42,
        verbose=False,
    )
    with pytest.raises(ValueError):
        exp._check_model_type("unsupported")


def test_prepare_experiment_input_scaling_and_graph(monkeypatch, sample_csv, tmp_path):
    # Return a sentinel graph without hitting filesystem
    sentinel_graph = object()
    monkeypatch.setattr(notebook.utils, "graph_from_dot_file", lambda path: sentinel_graph)
    exp = notebook.Experiment(
        experiment_name="exp",
        csv_filename=str(sample_csv),
        dot_filename=str(tmp_path / "graph.dot"),
        model_type="nn",
        input_path=str(tmp_path),
        output_path=str(tmp_path),
        train_size=2 / 3,
        random_state=0,
        verbose=False,
    )
    assert exp.ref_graph is sentinel_graph
    # train/test split respects size and scaling flag default (False)
    assert len(exp.train_data) + len(exp.test_data) == len(exp.data)


def test_create_estimator_unknown(monkeypatch, sample_csv, tmp_path):
    monkeypatch.setattr(notebook.utils, "graph_from_dot_file", lambda _: None)
    exp = notebook.Experiment(
        experiment_name="exp",
        csv_filename=str(sample_csv),
        dot_filename=str(tmp_path / "graph.dot"),
        model_type="nn",
        input_path=str(tmp_path),
        output_path=str(tmp_path),
    )
    assert exp.create_estimator("nonexistent", name="x") is None


def test_fit_and_predict_use_created_estimator(monkeypatch, sample_csv, tmp_path):
    monkeypatch.setattr(notebook.utils, "graph_from_dot_file", lambda _: None)
    exp = notebook.Experiment(
        experiment_name="exp",
        csv_filename=str(sample_csv),
        dot_filename=str(tmp_path / "graph.dot"),
        model_type="nn",
        input_path=str(tmp_path),
        output_path=str(tmp_path),
    )
    monkeypatch.setattr(exp, "create_estimator", lambda estimator_name, name, **kwargs: DummyEstimator(name=name, **kwargs))
    exp.fit(estimator_name="rex", pipeline="p")
    assert isinstance(getattr(exp, "rex"), DummyEstimator)
    assert exp.is_fitted is True
    # Predict delegates to estimator stored under estimator_name
    exp.predict(estimator="rex")
    assert getattr(exp, "rex").predict_args[0][0] is exp.train_data


def test_fit_predict_handles_missing_estimator(monkeypatch, sample_csv, tmp_path):
    monkeypatch.setattr(notebook.utils, "graph_from_dot_file", lambda _: None)
    exp = notebook.Experiment(
        experiment_name="exp",
        csv_filename=str(sample_csv),
        dot_filename=str(tmp_path / "graph.dot"),
        model_type="nn",
        input_path=str(tmp_path),
        output_path=str(tmp_path),
    )
    monkeypatch.setattr(exp, "create_estimator", lambda *_, **__: None)
    with pytest.raises(ValueError):
        exp.fit_predict(estimator="unknown")


def test_load_and_save_roundtrip(monkeypatch, sample_csv, tmp_path):
    # Use a dummy estimator object and simulate persistence helpers
    dummy = DummyEstimator()
    monkeypatch.setattr(notebook, "Rex", DummyEstimator)
    monkeypatch.setattr(notebook.utils, "graph_from_dot_file", lambda _: None)
    exp = notebook.Experiment(
        experiment_name="exp",
        csv_filename=str(sample_csv),
        dot_filename=str(tmp_path / "graph.dot"),
        model_type="nn",
        input_path=str(tmp_path),
        output_path=str(tmp_path),
    )

    saved = {}
    monkeypatch.setattr(
        notebook.utils,
        "save_experiment",
        lambda name, folder, obj, overwrite=False: saved.update({"name": name, "folder": folder, "obj": obj}) or "path",
    )
    monkeypatch.setattr(
        notebook.utils,
        "load_experiment",
        lambda name, folder: dummy,
    )
    # Pretend estimator is already fitted
    exp.estimator_name = "rex"
    setattr(exp, "rex", dummy)

    path = exp.save(overwrite=True)
    assert path == "path"
    assert saved["obj"] is dummy

    loaded = exp.load()
    assert loaded.estimator_name == "rex"
    assert loaded.estimator is dummy
