"""
Well log style multi-track charts using Qt Charts.

This module provides two reusable widgets:
- TrackChart: a single vertical-track chart (wraps QChart)
- MultiTrackWidget: a horizontal container that lays out multiple TrackCharts

The demo at the bottom shows how to create several tracks side-by-side with a
shared depth (Y) range and independent X ranges/data.
"""

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtGui import QPainter
from PySide6.QtCore import Qt, QPointF, Signal, QObject
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QSizePolicy, QScrollArea, QVBoxLayout
import math
import sys
from typing import Sequence, Union

import numpy as np
import pandas as pd

class TrackChart(QChart):

    """Single track chart suitable for well-log style plotting.

    Features:
    - One QLineSeries (can be replaced with multiple if needed)
    - Independent X axis per track 
    - Y axis (depth) can be synchronized with other TrackChart instances
    """

    def __init__(self, title: str = "", x_min: float = 0.0, x_max: float = 1.0,
                 depth_min: float = 0.0, depth_max: float = 10000, show_y_axis: bool = True):
        super().__init__()
        self.setTitle(title)

        # axes
        self.axisX = QValueAxis()
        self.axisY = QValueAxis()

        self.addAxis(self.axisX, Qt.AlignTop)
        self.addAxis(self.axisY, Qt.AlignLeft)

        self.axisX.setRange(x_min, x_max)

        # series handling
        self._series_list = []
        self.series = None
        self._ensure_single_series()

        # depth: top (small) -> bottom (large). Default Qt charts have y increasing upward,
        # to make depth increase downward we reverse the axis if available.
        try:
            # QValueAxis may support setReverse in some Qt versions
            self.axisY.setRange(depth_min, depth_max)
            self.axisY.setReverse(True)
        except Exception:
            # Fallback: set range and rely on coordinated drawing (may show inverted)
            self.axisY.setRange(depth_min, depth_max)

        # grid and ticks
        self.axisY.setTickCount(11)
        self.axisX.setTickCount(4)
        # 刻度保留三位小数
        self.axisX.setLabelFormat("%.3f")
        self.axisY.setLabelFormat("%.3f")
        # y轴网格线
        self.axisY.setGridLineVisible(False)

        if not show_y_axis:
            self.axisY.setVisible(False)

        # Default series pen/visuals can be customized by caller

    def _clear_series(self):
        for series in self._series_list:
            self.removeSeries(series)
        self._series_list.clear()
        self.series = None

    def _add_series(self, series: QLineSeries):
        self.addSeries(series)
        series.attachAxis(self.axisX)
        series.attachAxis(self.axisY)
        self._series_list.append(series)
        if self.series is None:
            self.series = series
        return series

    def _ensure_single_series(self):
        if len(self._series_list) != 1:
            self._clear_series()
            series = QLineSeries()
            self._add_series(series)
        else:
            series = self._series_list[0]
        self.series = series
        self.legend().hide()
        return series

    def set_data(self, x: np.ndarray, y: np.ndarray):
        """Replace the series data. x and y must be 1D and same length."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        series = self._ensure_single_series()
        points = [QPointF(float(xi), float(yi)) for xi, yi in zip(x, y)]
        series.replace(points)

    def set_x_range(self, xmin: float, xmax: float):
        self.axisX.setRange(xmin, xmax)

    def set_depth_range(self, dmin: float, dmax: float):
        # try reversing to make depth increase downward
        try:
            self.axisY.setRange(dmin, dmax)
            self.axisY.setReverse(True)
        except Exception:
            self.axisY.setRange(dmin, dmax)

    def set_dataframe(self, df: pd.DataFrame, value_columns: Union[str, Sequence[str]], depth_column: str):
        """Plot one or more columns from *df* against *depth_column*."""
        if isinstance(value_columns, str):
            columns = [value_columns]
        else:
            columns = list(value_columns)
        if not columns:
            raise ValueError("value_columns must contain at least one column name")

        required = [depth_column, *columns]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in DataFrame: {missing}")

        depth_series = pd.to_numeric(df[depth_column], errors="coerce")
        valid_depth = depth_series.dropna()
        if valid_depth.empty:
            raise ValueError("No valid depth values available for plotting")

        self._clear_series()
        combined_x = []

        for col in columns:
            value_series = pd.to_numeric(df[col], errors="coerce")
            paired = pd.DataFrame({"depth": depth_series, "value": value_series}).dropna()
            if paired.empty:
                continue
            series = QLineSeries()
            series.setName(col)
            points = [QPointF(float(v), float(d)) for v, d in zip(paired["value"], paired["depth"])]
            self._add_series(series)
            series.replace(points)
            combined_x.extend(paired["value"].tolist())

        if not self._series_list:
            raise ValueError("No valid data found for requested columns")

        self.series = self._series_list[0]
        self.legend().setVisible(len(self._series_list) > 1)

        dmin = float(valid_depth.min())
        dmax = float(valid_depth.max())
        self.set_depth_range(dmin, dmax)

        x_values = [float(x) for x in combined_x if not math.isnan(float(x)) and math.isfinite(float(x))]
        if x_values:
            xmin = min(x_values)
            xmax = max(x_values)
            if math.isclose(xmin, xmax):
                pad = abs(xmin) * 0.01 or 1.0
                xmin -= pad
                xmax += pad
            self.axisX.setRange(xmin, xmax)


class MultiTrackWidget(QWidget):
    """Container widget that lays out multiple TrackChart instances horizontally.
    Usage:
        mt = MultiTrackWidget(depth_range=(0, 100))
        mt.add_track('GR', width=120)
        mt.add_track('Resistivity', width=140)
        mt.set_depth_range(0, 200)
    """

    def __init__(self, depth_range=(0.0, 100.0), parent=None):
        super().__init__(parent)
        self._depth_min, self._depth_max = depth_range

        # Outer layout holds the scroll area
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Container widget inside the scroll area - this holds the horizontal track views
        self._tracks_container = QWidget()
        self._tracks_layout = QHBoxLayout(self._tracks_container)
        self._tracks_layout.setContentsMargins(0, 0, 0, 0)
        self._tracks_layout.setSpacing(0)
        self._total_width = 0  # track the accumulated width so horizontal scroll can engage

        # Scroll area to allow horizontal scrolling when total width exceeds available space
        self._scroll = QScrollArea()
        # Let the inner widget resize to the viewport height; we will fix its minimum width for horizontal scrolling
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._tracks_container)

        self.layout.addWidget(self._scroll)

        # store tuples of (chart, view)
        self.tracks = []

    def add_track(self, title: str = "", width: int = 200, show_y_axis: bool = False,
                  x_range=(0.0, 1.0)):
        chart = TrackChart(title, x_min=x_range[0], x_max=x_range[1],
                           depth_min=self._depth_min, depth_max=self._depth_max,
                           show_y_axis=show_y_axis)
        chart.legend().hide()

        view = QChartView(chart)
        view.setRenderHint(QPainter.Antialiasing)
        # Fix the horizontal size so each track stays consistent and scroll works as widths accumulate
        view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        view.setFixedWidth(width)

        # add into the horizontal container inside the scroll area
        self._tracks_layout.addWidget(view)
        self.tracks.append((chart, view))
        self._update_container_width()
        return chart

    def set_depth_range(self, dmin: float, dmax: float):
        self._depth_min = dmin
        self._depth_max = dmax
        for chart, _ in self.tracks:
            chart.set_depth_range(dmin, dmax)

    def clear_tracks(self):
        while self.tracks:
            chart, view = self.tracks.pop()
            self._tracks_layout.removeWidget(view)
            view.setParent(None)
            view.deleteLater()
        self._update_container_width()

    def plot_dataframe(self, df: pd.DataFrame, track_specs, depth_column=None, widths=None):
        """Create tracks based on the provided *track_specs* sequence."""
        if depth_column is None:
            if not track_specs:
                raise ValueError("track_specs must contain at least the depth column")
            depth_column = track_specs[0]
            value_specs = track_specs[1:]
        else:
            value_specs = track_specs

        if not value_specs:
            raise ValueError("No value tracks specified")

        if isinstance(widths, int):
            resolved_widths = [widths] * len(value_specs)
        elif widths is None:
            resolved_widths = [200] * len(value_specs)
        else:
            resolved_widths = list(widths)
            if len(resolved_widths) != len(value_specs):
                raise ValueError("Length of widths must match number of value tracks")

        depth_series = pd.to_numeric(df[depth_column], errors="coerce").dropna()
        if depth_series.empty:
            raise ValueError("Depth column contains no valid numbers")

        dmin = float(depth_series.min())
        dmax = float(depth_series.max())
        self._depth_min = dmin
        self._depth_max = dmax

        self.clear_tracks()

        charts = []
        for idx, spec in enumerate(value_specs):
            if isinstance(spec, str):
                columns = spec
                title = spec
            else:
                columns = list(spec)
                if not columns:
                    continue
                title = " / ".join(columns)

            width = resolved_widths[idx]
            chart = self.add_track(title, width=width, show_y_axis=(idx == 0))
            chart.set_dataframe(df, columns, depth_column)
            charts.append(chart)

        return charts

    def _update_container_width(self):
        # Sum the fixed widths plus layout spacing to ensure the scroll area knows the real width
        spacing = self._tracks_layout.spacing() * max(len(self.tracks) - 1, 0)
        margins = self._tracks_layout.contentsMargins()
        margin_total = margins.left() + margins.right()
        content_width = sum(view.width() for _, view in self.tracks)
        self._tracks_container.setMinimumWidth(content_width + spacing + margin_total)
        # Ensure vertical fills the viewport height so content isn't clipped
        if self._scroll is not None and self._scroll.viewport() is not None:
            self._tracks_container.setMinimumHeight(self._scroll.viewport().height())
        self._tracks_container.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep container height in sync with scroll viewport to avoid vertical clipping
        if self._scroll is not None and self._scroll.viewport() is not None:
            self._tracks_container.setMinimumHeight(self._scroll.viewport().height())


def _demo():
    app = QApplication(sys.argv)

    # create multi-track widget and demo DataFrame
    mt = MultiTrackWidget()
    depth = np.linspace(1200, 1500, 300)
    df = pd.DataFrame({
        "MD": depth,
        "GR": 60 + 25 * np.sin(depth / 25),
        "RHOB": 2.3 + 0.05 * np.cos(depth / 35),
        "M2R1": 4 + np.sin(depth / 18),
        "M2R9": 6 + np.cos(depth / 22),
        "TVD": depth - 8 * np.sin(depth / 45),
    })

    tracks = ["MD", "GR", "RHOB", ["M2R1", "M2R9"], "TVD"]
    mt.plot_dataframe(df, tracks)

    mt.setWindowTitle("Multi-track well log demo")
    
    mt.resize(700, 800)
    mt.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()