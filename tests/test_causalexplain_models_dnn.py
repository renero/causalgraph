import pandas as pd
import pytest
import torch

from causalexplain.models import dnn


class DummyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return torch.zeros((x.shape[0], 1))


class DummyMLPModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = DummyTorchModel()
        self.trained = False

    def train(self):
        self.trained = True


def _small_frame():
    return pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})


def test_nnregressor_fit_predict_and_score(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    class FakeProgBar:
        def start_subtask(self, *_, **__):
            return self

        def update_subtask(self, *_, **__):
            return None

        def remove(self, *_, **__):
            return None

    monkeypatch.setattr(dnn, "ProgBar", FakeProgBar)

    df = _small_frame()
    rex = dnn.NNRegressor(
        hidden_dim=[2],
        activation="relu",
        num_epochs=1,
        batch_size=2,
        prog_bar=False,
        early_stop=False,
        min_delta=0.0,
    )
    rex.fit(df)

    preds = rex.predict(df)
    assert preds.shape[0] == len(df.columns)

    scores = rex.score(df)
    assert scores.shape[0] == len(df.columns)
    assert rex.is_fitted_ is True


def test_nnregressor_predict_requires_fit(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)
    rex = dnn.NNRegressor()
    with pytest.raises(AttributeError):
        rex.predict(_small_frame())


def test_nnregressor_drops_correlated_features(monkeypatch):
    monkeypatch.setattr(dnn, "MLPModel", DummyMLPModel)

    class FakeHierarchies:
        @staticmethod
        def compute_correlation_matrix(X):
            return X

        @staticmethod
        def compute_correlated_features(matrix, _th, feature_names, verbose=False):
            return {name: [f for f in feature_names if f != name] for name in feature_names}

    monkeypatch.setattr(dnn, "Hierarchies", FakeHierarchies)
    monkeypatch.setattr(dnn, "ProgBar", lambda *_, **__: None)

    df = _small_frame()
    rex = dnn.NNRegressor(correlation_th=0.5, prog_bar=False, early_stop=False, min_delta=0.0, num_epochs=1)
    rex.fit(df)

    # Each model should have received a single predictor after correlated removal.
    for target, model in rex.regressor.items():
        assert model.kwargs["input_size"] == 1
