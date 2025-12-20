from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class LogDataModel:
    """Model that stores logging data and metadata for the main view."""

    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        return self.data is not None and not self.data.empty

    def update(self, data: pd.DataFrame, metadata: Dict[str, object]) -> None:
        self.data = data
        self.metadata = metadata

    def clear(self) -> None:
        self.data = None
        self.metadata.clear()

    def depth_range(self) -> Tuple[float, float]:
        if not self.is_loaded:
            return 0.0, 0.0
        depth_series = self.data.iloc[:, 0]
        return float(depth_series.min()), float(depth_series.max())

    def subset_by_depth(self, start: float, end: float) -> pd.DataFrame:
        if not self.is_loaded:
            return pd.DataFrame()
        depth_column = self.data.columns[0]
        mask = self.data[depth_column].between(start, end)
        return self.data.loc[mask].copy()
