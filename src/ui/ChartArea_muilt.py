"""Compatibility wrapper that re-exports the pyqtgraph-based chart widgets."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from src.ui.ChartArea import MultiTrackWidget, TrackChart

__all__ = ["TrackChart", "MultiTrackWidget"]


def _demo() -> None:
    app = QApplication(sys.argv)

    mt = MultiTrackWidget()
    depth = np.linspace(1200, 1500, 300)
    df = pd.DataFrame(
        {
            "MD": depth,
            "GR": 60 + 25 * np.sin(depth / 25),
            "RHOB": 2.3 + 0.05 * np.cos(depth / 35),
            "M2R1": 4 + np.sin(depth / 18),
            "M2R9": 6 + np.cos(depth / 22),
            "TVD": depth - 8 * np.sin(depth / 45),
        }
    )

    tracks = ["MD", "GR", "RHOB", ["M2R1", "M2R9"], "TVD"]
    mt.plot_dataframe(df, tracks, depth_column="MD")

    mt.setWindowTitle("Multi-track well log demo (pyqtgraph)")
    mt.resize(700, 800)
    mt.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()