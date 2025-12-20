from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class DrillingParameters:
    Dh: float = 0.0
    Dhi: float = 0.0
    fai300: float = 0.0
    fai600: float = 0.0


@dataclass
class DrillingPressureModel:
    """Model that tracks drilling pressure calculation state."""

    main_data: Optional[pd.DataFrame] = None
    cf_data: Optional[Dict[int, tuple]] = None
    result: Optional[pd.DataFrame] = None
    params: DrillingParameters = field(default_factory=DrillingParameters)

    def is_ready(self) -> bool:
        return self.main_data is not None and self.cf_data is not None

    def clear_result(self) -> None:
        self.result = None
