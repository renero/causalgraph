import pandas as pd
import torch

from causalexplain.models._columnar import ColumnsDataset


def test_columns_dataset_shapes_and_length():
    df = pd.DataFrame(
        {
            "target": [0.1, 0.2, 0.3],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        }
    )

    dataset = ColumnsDataset("target", df)

    assert len(dataset) == 3
    assert dataset.features.dtype == torch.float32
    assert dataset.target.dtype == torch.float32

    features, target = dataset[1]
    assert torch.equal(target, torch.tensor([0.2], dtype=torch.float32))
    assert torch.allclose(features, torch.tensor([2.0, 5.0]))
