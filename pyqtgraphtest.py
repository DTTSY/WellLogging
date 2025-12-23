import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget, QSplitter
from PySide6.QtCore import Qt

class LogTrack(pg.PlotWidget):
    """单个测井道插件"""
    def __init__(self, title, unit, color, parent=None):
        super().__init__(parent)
        self.setClipToView(True)
        self.setDownsampling(mode='peak')
        pg.setConfigOptions(
    antialias=False,       # 测井曲线通常不需要抗锯齿，关闭可极大提升速度
    useOpenGL=True,        # 开启 OpenGL 硬件加速渲染
    enableExperimental=True # 启用实验性性能优化
)
        
        # UI设置
        self.setBackground('w')  # 白色背景更符合报告习惯
        self.getPlotItem().invertY(True)  # 深度向下增加
        self.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.setLabel('top', title, units=unit)
        self.getPlotItem().getAxis('bottom').setStyle(tickTextOffset=5)
        
        # 创建曲线对象
        self.curve = self.plot(pen=pg.mkPen(color=color, width=1.5))

    def update_data(self, depth, values):
        self.curve.setData(values, depth)

class MultiTrackView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多道测井数据同步分析系统")
        self.setMinimumSize(1200, 800)

        # 主布局使用 QSplitter，允许用户手动调整每道的宽度
        self.splitter = QSplitter(Qt.Horizontal)
        
        # 1. 创建三道：GR(伽马), RES(电阻率), POR(孔隙度)
        self.track_gr = LogTrack("GR", "gAPI", "#2ecc71")
        self.track_res = LogTrack("RES", "ohm.m", "#e74c3c")
        self.track_por = LogTrack("POR", "v/v", "#3498db")

        # 2. 关键：将三道的 Y 轴（深度轴）联动
        # 这样缩放其中一个，其他两个会同步滚动
        self.track_res.setYLink(self.track_gr)
        self.track_por.setYLink(self.track_gr)

        # 添加到布局
        self.splitter.addWidget(self.track_gr)
        self.splitter.addWidget(self.track_res)
        self.splitter.addWidget(self.track_por)

        self.setCentralWidget(self.splitter)
        
        # 模拟加载大数据
        self.load_well_data()

    def load_well_data(self):
        # 模拟 200,000 行数据
        n_points = 2000
        depth = np.linspace(0, 3000, n_points)
        
        # 模拟三种不同的物理特征
        gr_data = 50 + np.cumsum(np.random.randn(n_points) * 0.3)
        res_data = np.abs(10 + np.cumsum(np.random.randn(n_points) * 0.5))
        por_data = 0.3 - (np.abs(np.random.randn(n_points)) * 0.001)

        # 更新各道数据
        self.track_gr.update_data(depth, gr_data)
        self.track_res.update_data(depth, res_data)
        self.track_por.update_data(depth, por_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局风格
    pg.setConfigOptions(antialias=True, foreground='k', background='w')
    window = MultiTrackView()
    window.show()
    sys.exit(app.exec())