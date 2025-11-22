import pandas as pd
import torch
from torch.utils.data import Dataset


class ColumnsDataset(Dataset):
    def __init__(self, target_name, df: pd.DataFrame):
        target = df.loc[:, target_name].values.reshape(-1, 1)
        features = df.drop(target_name, axis=1).values
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # Align returned feature dtype with the current global default to avoid
        # downstream dtype mismatches (some tests change the default to double).
        features = self.features[idx].to(torch.get_default_dtype())
        return [features, self.target[idx]]
