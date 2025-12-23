"""
Well log style multi-track charts implemented with pyqtgraph.

This module provides two reusable widgets:
- TrackChart: a single vertical-track chart (wraps a pyqtgraph PlotWidget)
- MultiTrackWidget: a horizontal container that lays out multiple TrackCharts
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


pg.setConfigOptions(
    antialias=False,       # 测井曲线通常不需要抗锯齿，关闭可极大提升速度
    useOpenGL=True,        # 开启 OpenGL 硬件加速渲染
    enableExperimental=True, # 启用实验性性能优化
    )


def _pen_for_index(idx: int) -> pg.mkPen:
    """Return a colored pen from the pyqtgraph palette for consistent styling."""
    return pg.mkPen(pg.intColor(idx, hues=10, maxValue=220, minValue=120), width=1.5)


class TrackChart(pg.PlotWidget):
    """Single track chart suitable for well-log style plotting."""

    def __init__(
        self,
        title: str = "",
        x_min: float = 0.0,
        x_max: float = 1.0,
        depth_min: float = 0.0,
        depth_max: float = 10000,
        show_y_axis: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._curves: list[pg.PlotDataItem] = []
        self._legend: pg.LegendItem | None = None
        self._init_plot(title, x_min, x_max, depth_min, depth_max, show_y_axis)
        # UI设置
        self.setBackground('w')  # 白色背景更符合报告习惯
        self.setClipToView(True) # 提升大数据量渲染性能


    def _init_plot(
        self, title: str, x_min: float, x_max: float, depth_min: float, depth_max: float, show_y_axis: bool
    ) -> None:
        plot = self.getPlotItem()
        plot.setTitle(title)
        plot.invertY(True)  # depth increases downward
        # plot.setMenuEnabled(False)
        plot.showGrid(x=False, y=False)
        # plot.enableAutoRange(False, False)
        # plot.setXRange(x_min, x_max, padding=0)
        plot.setYRange(depth_min, depth_max, padding=0)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        self._ensure_single_curve()

        plot.showGrid(x=True, y=True, alpha=0.3)
        # self.setLabel('top', title, units=unit)
        plot.getAxis('bottom').setStyle(tickTextOffset=5)

    def _clear_curves(self) -> None:
        plot = self.getPlotItem()
        for curve in self._curves:
            plot.removeItem(curve)
        self._curves.clear()
        if self._legend is not None:
            plot.removeItem(self._legend)
            self._legend = None

    def _ensure_single_curve(self) -> pg.PlotDataItem:
        if len(self._curves) != 1:
            self._clear_curves()
            curve = self.getPlotItem().plot([], [], pen=_pen_for_index(0))
            self._curves = [curve]
        return self._curves[0]

    def set_data(self, x: np.ndarray, y: np.ndarray) -> None:
        """Replace the curve data. x and y must be 1D and the same length."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        curve = self._ensure_single_curve()
        curve.setData(x, y, connect="finite")

    def set_dataframe(self, df: pd.DataFrame, value_columns: Union[str, Sequence[str]], depth_column: str) -> None:
        """Plot one or more columns from *df* against *depth_column*."""
        if isinstance(value_columns, str):
            columns: Iterable[str] = [value_columns]
        else:
            columns = list(value_columns)
        if not columns:
            raise ValueError("value_columns must contain at least one column name")

        missing = [col for col in [depth_column, *columns] if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in DataFrame: {missing}")

        depth_series = pd.to_numeric(df[depth_column], errors="coerce").dropna()
        if depth_series.empty:
            raise ValueError("Depth column contains no valid numbers")

        self._clear_curves()
        plot = self.getPlotItem()
        x_values: list[float] = []

        for idx, col in enumerate(columns):
            series = pd.to_numeric(df[col], errors="coerce")
            paired = pd.DataFrame({"depth": depth_series, "value": series}).dropna()
            if paired.empty:
                continue
            curve = plot.plot([], [], pen=_pen_for_index(idx))
            curve.setData(paired["value"].to_numpy(), paired["depth"].to_numpy(), connect="finite")
            self._curves.append(curve)
            x_values.extend(paired["value"].tolist())

        if not self._curves:
            raise ValueError("No valid data found for requested columns")

        if len(self._curves) > 1:
            self._legend = plot.addLegend(offset=(10, 10))
            for curve, col in zip(self._curves, columns):
                self._legend.addItem(curve, name=str(col))

        dmin = float(depth_series.min())
        dmax = float(depth_series.max())
        self.set_depth_range(dmin, dmax)

        finite_x = [float(x) for x in x_values if np.isfinite(x)]
        if finite_x:
            xmin, xmax = min(finite_x), max(finite_x)
            if np.isclose(xmin, xmax):
                pad = abs(xmin) * 0.01 or 1.0

    # def resizeEvent(self, event) -> None:  # type: ignore[override]
    #     super().resizeEvent(event)
    #     if self.parent._scroll.viewport() is not None:
    #         self.parent._splitter_container.setMinimumHeight(self.parent._scroll.viewport().height())
    #         xmin -= pad
    #         xmax += pad
    #         pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    #         self.set_x_range(xmin - pad, xmax + pad)

    def set_x_range(self, xmin: float, xmax: float) -> None:
        self.getPlotItem().setXRange(xmin, xmax, padding=0)

    def set_depth_range(self, dmin: float, dmax: float) -> None:
        plot = self.getPlotItem()
        plot.setYRange(dmin, dmax, padding=0)
        plot.invertY(True)

    def link_y(self, other: "TrackChart") -> None:
        self.getPlotItem().setYLink(other.getPlotItem())


class MultiTrackWidget(QWidget):
    """Container widget that lays out multiple TrackChart instances horizontally."""

    def __init__(
        self,
        depth_range=(0.0, 100.0),
        parent: QWidget | None = None,
        *,
        min_track_width: int = 120,
    ):
        super().__init__(parent)
        self._depth_min, self._depth_max = depth_range
        self._y_anchor: TrackChart | None = None
        self._min_track_width = max(300, int(min_track_width))

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self._splitter_container = QWidget()
        self._container_layout = QHBoxLayout(self._splitter_container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.splitterMoved.connect(self._on_splitter_moved)

        self._container_layout.addWidget(self._splitter)
        self._splitter_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._splitter_container)

        self.layout.addWidget(self._scroll)

        self.tracks: list[TrackChart] = []
        self._requested_sizes: list[int] = []

    def add_track(
        self,
        title: str = "",
        width: int = 300,
        show_y_axis: bool = False,
        x_range: tuple[float, float] = (0.0, 1.0),
    ) -> TrackChart:
        chart = TrackChart(
            title,
            x_min=x_range[0],
            x_max=x_range[1],
            depth_min=self._depth_min,
            depth_max=self._depth_max,
            show_y_axis=show_y_axis,
            parent=self,
        )

        if self._y_anchor is None:
            self._y_anchor = chart
        else:
            chart.link_y(self._y_anchor)

        desired_width = max(width, self._min_track_width)
        chart.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        chart.setMinimumWidth(self._min_track_width)

        self._splitter.addWidget(chart)
        self.tracks.append(chart)
        self._requested_sizes.append(desired_width)
        self._apply_splitter_sizes(force=True)
        self._update_container_width()
        return chart

    def set_depth_range(self, dmin: float, dmax: float) -> None:
        self._depth_min = dmin
        self._depth_max = dmax
        for chart in self.tracks:
            chart.set_depth_range(dmin, dmax)

    def clear_tracks(self) -> None:
        while self.tracks:
            chart = self.tracks.pop()
            chart.setParent(None)
            chart.deleteLater()
        self._y_anchor = None
        self._requested_sizes.clear()
        self._apply_splitter_sizes(force=True)
        self._update_container_width()

    def plot_dataframe(
        self,
        df: pd.DataFrame,
        track_specs: Sequence[Union[str, Sequence[str]]],
        depth_column: str,
        widths: Union[int, Sequence[int]] | None = None,
    ) -> list[TrackChart]:
        """Create tracks based on DataFrame column specs.

        track_specs examples:
            ["MD", "GR", "RHOB", ["M2R1", "M2R9"]]
        """
        if not track_specs:
            raise ValueError("track_specs must contain at least one track")

        if isinstance(widths, int):
            resolved_widths = [widths] * len(track_specs)
        elif widths is None:
            resolved_widths = [self._min_track_width] * len(track_specs)
        else:
            resolved_widths = list(widths)
            if len(resolved_widths) != len(track_specs):
                raise ValueError("Length of widths must match number of tracks")

        depth_series = pd.to_numeric(df[depth_column], errors="coerce").dropna()
        if depth_series.empty:
            raise ValueError("Depth column contains no valid numbers")

        self._depth_min = float(depth_series.min())
        self._depth_max = float(depth_series.max())

        self.clear_tracks()
        charts: list[TrackChart] = []

        for idx, spec in enumerate(track_specs):
            if isinstance(spec, str):
                columns: Union[str, Sequence[str]] = spec
                title = spec
            else:
                columns = list(spec)
                if not columns:
                    continue
                title = " / ".join(columns)

            chart = self.add_track(title, width=resolved_widths[idx], show_y_axis=(idx == 0))
            chart.set_dataframe(df, columns, depth_column)
            charts.append(chart)
        return charts

    def _on_splitter_moved(self, pos: int, index: int) -> None:  # pragma: no cover - UI callback
        self._requested_sizes = [max(self._min_track_width, size) for size in self._splitter.sizes()]
        self._update_container_width()

    def _apply_splitter_sizes(self, *, force: bool = False) -> None:
        if not self.tracks:
            return
        sizes = self._splitter.sizes()
        needs_update = force or len(sizes) != len(self.tracks) or any(size <= 0 for size in sizes)
        if not needs_update:
            return
        new_sizes: list[int] = []
        for idx in range(len(self.tracks)):
            if idx < len(self._requested_sizes):
                new_sizes.append(max(self._min_track_width, self._requested_sizes[idx]))
            else:
                new_sizes.append(self._min_track_width)
        self._splitter.setSizes(new_sizes)

    def _update_container_width(self) -> None:
        if not self.tracks:
            self._splitter_container.setMinimumWidth(0)
            return

        sizes = self._splitter.sizes()
        if len(sizes) != len(self.tracks):
            sizes = self._requested_sizes or [self._min_track_width] * len(self.tracks)

        handle_extra = self._splitter.handleWidth() * max(len(self.tracks) - 1, 0)
        total_width = sum(max(self._min_track_width, size) for size in sizes) + handle_extra

        margins = self._container_layout.contentsMargins()
        total_width += margins.left() + margins.right()

        self._splitter_container.setMinimumWidth(total_width)
        if self._scroll.viewport() is not None:
            self._splitter_container.setMinimumHeight(self._scroll.viewport().height())



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
    mt.plot_dataframe(df, tracks, depth_column="MD" ,widths=300)

    mt.setWindowTitle("Multi-track well log demo (pyqtgraph)")
    mt.resize(800, 800)
    mt.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()