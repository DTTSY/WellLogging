from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from src.models.log_data_model import LogDataModel


class LogDataController:
    """Controller responsible for loading and slicing logging data."""

    def __init__(self, model: LogDataModel):
        self._model = model

    @property
    def model(self) -> LogDataModel:
        return self._model

    def load_file(self, file_path: str) -> None:
        data = pd.read_csv(file_path)
        well_name = os.path.splitext(os.path.basename(file_path))[0]
        depth_range = self._calculate_depth_range(data)
        metadata = {"well_name": well_name, "depth_range": depth_range, "file_path": file_path}
        self._model.update(data, metadata)

    def depth_defaults(self) -> Tuple[float, float]:
        return self._model.depth_range()

    def filtered_data(self, start_depth: float, end_depth: float) -> pd.DataFrame:
        return self._model.subset_by_depth(start_depth, end_depth)

    def _calculate_depth_range(self, data: pd.DataFrame) -> Tuple[float, float]:
        depth_series = data.iloc[:, 0]
        return float(depth_series.min()), float(depth_series.max())
