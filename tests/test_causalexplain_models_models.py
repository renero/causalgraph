import pandas as pd
import torch

from causalexplain.models import _models
from causalexplain.models._models import BaseModel, MLPModel


class DummyLogger:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.logger = kwargs.get("logger")
        self.fit_called = False

    def fit(self, *_, **__):
        self.fit_called = True
        return None


def _small_dataframe():
    return pd.DataFrame(
        {
            "target": [0.0, 1.0, 2.0, 3.0, 4.0],
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        }
    )


def test_base_model_initializes_loaders_and_batch(monkeypatch):
    monkeypatch.setattr(_models, "TensorBoardLogger", DummyLogger)
    df = _small_dataframe()
    base = BaseModel(
        target="target",
        dataframe=df,
        test_size=0.6,
        batch_size=4,
        tb_suffix="test",
        seed=0,
        early_stop=False,
    )

    # Validation split was smaller than requested batch size, so it shrinks.
    assert base.batch_size == 3
    assert len(base.train_loader.dataset) + len(base.val_loader.dataset) == len(df)


def test_override_extras_allows_custom_values(monkeypatch):
    monkeypatch.setattr(_models, "TensorBoardLogger", DummyLogger)
    df = _small_dataframe()
    base = BaseModel(
        target="target",
        dataframe=df,
        test_size=0.5,
        batch_size=2,
        tb_suffix="test",
        seed=0,
        early_stop=False,
    )
    base.n_rows = 12
    base.batch_size = 3

    base.override_extras(log_every_n_steps=7, fast_dev_run=True)
    assert base.extra_trainer_args["log_every_n_steps"] == 7
    assert base.extra_trainer_args["fast_dev_run"] is True
    assert base.extra_trainer_args["enable_model_summary"] is False


def test_mlp_model_initializes_trainer(monkeypatch):
    monkeypatch.setattr(_models, "TensorBoardLogger", DummyLogger)
    monkeypatch.setattr(_models, "Trainer", DummyTrainer)
    df = _small_dataframe()

    model = MLPModel(
        target="target",
        input_size=df.shape[1],
        hidden_dim=[2],
        activation="relu",
        learning_rate=0.01,
        batch_size=2,
        loss_fn="mse",
        dropout=0.0,
        num_epochs=1,
        dataframe=df,
        test_size=0.4,
        device="cpu",
        seed=0,
        early_stop=False,
        log_every_n_steps=5,
    )

    assert isinstance(model.trainer, DummyTrainer)
    assert model.trainer.kwargs["accelerator"] == "cpu"
    assert model.trainer.kwargs["log_every_n_steps"] == 5


def test_init_callbacks_adds_progress_bar(monkeypatch):
    monkeypatch.setattr(_models, "TensorBoardLogger", DummyLogger)

    class DummyBar:
        def init_train_tqdm(self):
            return type("TQDM", (), {"dynamic_ncols": False, "ncols": 0})()

    monkeypatch.setattr(_models, "TQDMProgressBar", DummyBar)
    df = _small_dataframe()
    base = BaseModel(
        target="target",
        dataframe=df,
        test_size=0.5,
        batch_size=2,
        tb_suffix="test",
        seed=0,
    )
    base.init_callbacks(early_stop=True, prog_bar=True)
    bar = next(cb for cb in base.callbacks if isinstance(cb, DummyBar))
    result = bar.init_train_tqdm()
    assert getattr(result, "ncols", None) == 0
    assert any(cb.__class__.__name__ == "EarlyStopping" for cb in base.callbacks)


def test_mlp_model_train_invokes_trainer(monkeypatch):
    monkeypatch.setattr(_models, "TensorBoardLogger", DummyLogger)
    dummy_trainer = DummyTrainer()
    monkeypatch.setattr(_models, "Trainer", lambda *a, **k: dummy_trainer)
    df = _small_dataframe()

    model = MLPModel(
        target="target",
        input_size=df.shape[1],
        hidden_dim=[2],
        activation="relu",
        learning_rate=0.01,
        batch_size=2,
        loss_fn="mse",
        dropout=0.0,
        num_epochs=1,
        dataframe=df,
        test_size=0.4,
        device="cpu",
        seed=0,
    )

    model.train()
    assert dummy_trainer.fit_called is True
