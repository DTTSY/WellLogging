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
import sys
import numpy as np


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

        # series
        self.series = QLineSeries()
        self.addSeries(self.series)

        # axes
        self.axisX = QValueAxis()
        self.axisY = QValueAxis()

        self.addAxis(self.axisX, Qt.AlignBottom)
        self.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

        self.axisX.setRange(x_min, x_max)

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
        # 刻度保留两位小数
        self.axisX.setLabelFormat("%.2f")
        self.axisY.setLabelFormat("%.2f")
        # y轴网格线
        self.axisY.setGridLineVisible(False)

        if not show_y_axis:
            self.axisY.setVisible(False)

        # Default series pen/visuals can be customized by caller

    def set_data(self, x: np.ndarray, y: np.ndarray):
        """Replace the series data. x and y must be 1D and same length."""
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        points = [QPointF(float(xi), float(yi)) for xi, yi in zip(x, y)]
        self.series.replace(points)

    def set_x_range(self, xmin: float, xmax: float):
        self.axisX.setRange(xmin, xmax)

    def set_depth_range(self, dmin: float, dmax: float):
        # try reversing to make depth increase downward
        try:
            self.axisY.setRange(dmin, dmax)
            self.axisY.setReverse(True)
        except Exception:
            self.axisY.setRange(dmin, dmax)


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
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(0)

        # Container widget inside the scroll area - this holds the horizontal track views
        self._tracks_container = QWidget()
        self._tracks_layout = QHBoxLayout(self._tracks_container)
        self._tracks_layout.setContentsMargins(0, 0, 0, 0)
        self._tracks_layout.setSpacing(6)
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

    # create multi-track widget
    mt = MultiTrackWidget(depth_range=(0, 100))

    # create 5 tracks with sample data
    n_tracks = 5
    depths = np.linspace(0, 100, 500)

    titles = [f"Track {i+1}" for i in range(n_tracks)]
    widths = [100, 110, 110, 110, 100]

    for i, (t, w) in enumerate(zip(titles, widths)):
        chart = mt.add_track(t, width=w, show_y_axis=(i == 0), x_range=(0, 1))
        # sample data: a shifted sine + noise per track
        x = 0.5 + 0.4 * np.sin(2 * np.pi * (depths / 100.0) * (i + 1))
        x += 0.05 * np.random.randn(depths.size)
        chart.set_data(x, depths)

    mt.setWindowTitle("Multi-track well log demo")
    
    mt.resize(700, 800)
    mt.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()