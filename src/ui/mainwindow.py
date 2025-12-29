from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QLabel,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QDoubleSpinBox,
    QWidget,
    QProgressDialog,
)
from PySide6.QtGui import QImage, QPixmap
import pandas as pd
import os

from src.controllers.log_data_controller import LogDataController
from src.controllers.drilling_pressure_controller import (
    DrillingPressureController,
)
from src.models.log_data_model import LogDataModel
from src.models.drilling_pressure_model import DrillingPressureModel
from src.ui.ChartArea import MultiTrackWidget
from src.ui.DrillingPressureCalculator_window_ui import Ui_w_DrillingPressureCalculator
from src.core.algorithm.DrillingPressureCalculator import OperationType


class DrillingPressureCalculator_ui(QWidget, Ui_w_DrillingPressureCalculator):
    def __init__(self, parent=None, controller: DrillingPressureController | None = None):
        super(DrillingPressureCalculator_ui, self).__init__(parent)
        self.setupUi(self)
        self.controller = controller or DrillingPressureController(DrillingPressureModel())
        self._model = self.controller.model
        self.pb_open_main.clicked.connect(lambda: self.open_file("main_data"))
        self.pb_open_dpf.clicked.connect(lambda: self.open_file("cf_data"))
        self._calc_thread = None
        self._worker = None
        self._progress_dialog = None

        self.pb_up.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_OUT))
        self.pb_down.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_IN))

    def handle_move(self, operation: OperationType):
        if self.data_validation():
            self.drilling_action(operation=operation)

    def data_validation(self):
        if self._model.main_data is None:
            QMessageBox.warning(self, "数据缺失", "请先导入主数据文件")
            return False
        if self._model.cf_data is None:
            QMessageBox.warning(self, "数据缺失", "请先导入坍塌破裂压力数据文件")
            return False
        self.controller.update_parameters(
            Dh=self.dsb_Dh.value(),
            Dhi=self.dsb_Dhi.value(),
            fai300=self.dsb_fai300.value(),
            fai600=self.dsb_fai600.value(),
        )
        return True
    
    def drilling_action(self, operation: OperationType):
        if self._calc_thread is not None:
            QMessageBox.information(self, "计算进行中", "请等待当前计算完成。")
            return

        if self._model.main_data is None or self._model.cf_data is None:
            QMessageBox.warning(self, "数据缺失", "请先导入所需的数据文件。")
            return

        self.pb_up.setEnabled(False)
        self.pb_down.setEnabled(False)
        self._progress_dialog = QProgressDialog("准备计算参数", None, 0, 100, self)
        self._progress_dialog.setWindowTitle("计算中")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setCancelButton(None)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.setAutoReset(False)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.show()

        thread = QThread(self)
        try:
            worker = self.controller.create_worker(operation)
        except ValueError as exc:
            self._reset_progress_state()
            self.pb_up.setEnabled(True)
            self.pb_down.setEnabled(True)
            QMessageBox.warning(self, "数据缺失", str(exc))
            return

        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_progress_update)
        worker.finished.connect(self._handle_calculation_finished)
        worker.error.connect(self._handle_calculation_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_calculation_thread_finished)

        self._calc_thread = thread
        self._worker = worker
        thread.start()

    def _handle_progress_update(self, value: int, message: str, elapsed: float):
        if self._progress_dialog is None:
            return
        self._progress_dialog.setValue(value)
        self._progress_dialog.setLabelText(f"{message}\n耗时: {elapsed:.1f} 秒")

    def _handle_calculation_finished(self, df: pd.DataFrame, elapsed: float):
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(100)
            self._progress_dialog.setLabelText(f"已保存结果\n总耗时: {elapsed:.2f} 秒")
            self._progress_dialog.close()
            self._progress_dialog = None
        self.pb_up.setEnabled(True)
        self.pb_down.setEnabled(True)
        QMessageBox.information(self, "计算完成", f"计算耗时: {elapsed:.2f} 秒，钻井压力计算已完成并保存结果文件。")

    def _handle_calculation_error(self, message: str):
        self._reset_progress_state()
        self.pb_up.setEnabled(True)
        self.pb_down.setEnabled(True)
        QMessageBox.critical(self, "计算失败", message)

    def _on_calculation_thread_finished(self):
        self._calc_thread = None
        self._worker = None

    def _reset_progress_state(self) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None

    def open_file(self, file_type: str):
        fname, _ = QFileDialog.getOpenFileName(self, "Open main data file", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if not fname:
            QMessageBox.information(self, "No File Selected", "No file was selected.")
            return
        try:
            if file_type == "main_data":
                self.controller.load_main_data(fname)
            elif file_type == "cf_data":
                self.controller.load_cf_data(fname)
            QMessageBox.information(self, "File Loaded", f"Successfully loaded file: {os.path.basename(fname)}")
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Failed to load file:\n{e}")
            return

class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app #declare an app member
        self.setWindowTitle("井壁稳定优化系统")
        # 设置窗口的最小尺寸
        self.setMinimumSize(QSize(1200, 800))
        self.mt = None
        self.log_model = LogDataModel()
        self.log_controller = LogDataController(self.log_model)

        #Menubar and menus
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        # new_action =  file_menu.addAction("New")
        open_action = file_menu.addAction("导入 地质力学数据")

        open_d_action = file_menu.addAction("导入 钻井动态数据")
        open_d_action.triggered.connect(lambda: QMessageBox.warning(self, "导入 钻井动态数据", "方法未实现"))
        # open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a CSV/text file (value, depth)")

        def open_file():
            fname, _ = QFileDialog.getOpenFileName(self, "Open data file", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
            if not fname:
                return
            try:
                # data = np.loadtxt(fname, delimiter=',')
                self.log_controller.load_file(fname)
                data = self.log_model.data
                self.statusBar().showMessage(f"Loaded file: {os.path.basename(fname)}", 5000)
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Failed to load file:\n{e}")
                return
            # Expect at least two columns: value, depth
            if data.ndim == 1:
                QMessageBox.warning(self, "Format error", "File must contain two columns (value, depth)")
                return
            if data.shape[1] < 2:
                QMessageBox.warning(self, "Format error", "File must have at least two columns (value, depth)")
                return
            self.set_mt()

        open_action.triggered.connect(open_file)


        edit_menu =menu_bar.addMenu("稳定性分析")
        edit_menu.addAction("地质力学静态计算").triggered.connect(lambda: QMessageBox.warning(self, "地质力学静态计算", "方法未实现"))
        edit_menu.addAction("压力波动动态计算").triggered.connect(self.open_drilling_pressure_calculator)
        edit_menu.addAction("机械扰动动态计算").triggered.connect(lambda: QMessageBox.warning(self, "机械扰动动态计算", "方法未实现"))
        # edit_menu.addAction("Undo")
        # edit_menu.addAction("Redo")

        #A bunch of other menu options just for the fun of it
        # menu_bar.addMenu("Window")
        settings_menu = menu_bar.addMenu("设置")
        settings_menu_action = settings_menu.addAction("首选项")
        settings_menu_action.triggered.connect(lambda: QMessageBox.warning(self, "首选项", "方法未实现"))
        menu_bar.addMenu("帮助")



        #Working with toolbars
        toolbar = QToolBar("My main toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setAllowedAreas(Qt.LeftToolBarArea)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setMinimumWidth(100)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        #Add the quit action to the toolbar
        # toolbar.addAction(quit_action)

        action1 = QAction("调整井深", self)
        action1.setStatusTip("Status message for some action")
        action1.triggered.connect(self.adjustDepthValue)
        toolbar.addAction(action1)

        # action2 = QAction(QIcon("start.png"), "Some other action", self)
        # action2.setStatusTip("Status message for some other action")
        # action2.triggered.connect(self.toolbar_button_click)
        # #action2.setCheckable(True)
        # toolbar.addAction(action2)

        # toolbar.addSeparator()
        # toolbar.addWidget(QPushButton("Click here"))


        # Working with status bars
        StatusBar = QStatusBar(self)
        StatusBar.showMessage("Ready")
        self.setStatusBar(StatusBar)

        # mt = MultiTrackWidget(depth_range=(0, 100))
        # self.mt = mt

        # # create 5 tracks with sample data
        # n_tracks = 5
        # depths = np.linspace(0, 100, 500)

        # titles = [f"Track {i+1}" for i in range(n_tracks)]

        # for i, t in enumerate(titles):
        #     chart = mt.add_track(t, width=300, show_y_axis=(i == 0), x_range=(0, 1))
        #     # sample data: a shifted sine + noise per track
        #     x = 0.5 + 0.4 * np.sin(2 * np.pi * (depths / 100.0) * (i + 1))
        #     x += 0.05 * np.random.randn(depths.size)
        #     chart.set_data(x, depths)
        # # mt.resize(700, 800)

        # # Place the multi-track widget in the main window
        # self.setCentralWidget(mt)
        # 显示图片
        image = QImage("assets/images/start.jpg")
        self.setCentralWidget(QLabel(pixmap=QPixmap.fromImage(image)))

    def open_drilling_pressure_calculator(self):
        model = DrillingPressureModel()
        model.cf_data = self.wel
        controller = DrillingPressureController(model)
        self.dlg = DrillingPressureCalculator_ui(controller=controller)
        self.dlg.show()

    def quit_app(self):
        self.app.quit()

    def toolbar_button_click(self):
        self.statusBar().showMessage("Message from my app",3000)
    
    def adjustDepthValue(self):
        # 打开一个对话课框，获取用户输入的深度值，最大值和最小值
        if not self.log_model.is_loaded:
            QMessageBox.warning(self, "数据缺失", "未导入有效数据")
            return

        # determine sensible defaults
        default_min, default_max = self.log_controller.depth_defaults()

        dlg = QDialog(self)
        dlg.setWindowTitle("调整深度范围")
        layout = QFormLayout(dlg)

        spin_min = QDoubleSpinBox(dlg)
        spin_min.setDecimals(2)
        spin_min.setRange(-1e12, 1e12)
        spin_min.setValue(default_min)

        spin_max = QDoubleSpinBox(dlg)
        spin_max.setDecimals(2)
        spin_max.setRange(-1e12, 1e12)
        spin_max.setValue(default_max)

        layout.addRow("起始深度:", spin_min)
        layout.addRow("终止深度:", spin_max)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addRow(buttons)

        ok = dlg.exec() == QDialog.Accepted
        dmin = float(spin_min.value())
        dmax = float(spin_max.value())
        if not ok:
            return
        if dmin >= dmax:
            QMessageBox.warning(self, "深度范围错误", "起始深度需小于终止深度")
            return

        try:
            self.set_mt(depth_range=(dmin, dmax))
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply depth range:\n{e}")
        
    
    def set_mt(self, depth_range=None):
        if self.mt is not None:
            self.mt.setParent(None)  # Remove existing widget
            self.mt.deleteLater()
            self.mt = None
        if not self.log_model.is_loaded:
            return
        data = self.log_model.data
        if depth_range is not None:
            dmin, dmax = depth_range
            data = self.log_controller.filtered_data(dmin, dmax)
        else:
            dmin = data.iloc[:, 0].min()
            dmax = data.iloc[:, 0].max()
        mt = MultiTrackWidget(depth_range=(dmin, dmax))
        self.mt = mt

        depths = data.iloc[:, 0].to_numpy()

        titles = [name for name in data.columns[1:]]

        for i, t in enumerate(titles):
            w=300
            chart = mt.add_track(t, width=w, show_y_axis=(i == 0), x_range=(0, data[t].max()))
            # sample data: a shifted sine + noise per track
            x = data[t].to_numpy()
            chart.set_data(x, depths)
        # mt.resize(700, 800)
        # 固定子图之间的水平间隔（像素）
        fixed_gap = 2
        mt._tracks_layout.setSpacing(fixed_gap)

        # Place the multi-track widget in the main window
        self.setCentralWidget(mt)
        metadata = self.log_model.metadata
        self.statusBar().showMessage(f"井名: {metadata.get('well_name', '未知')}  深度范围: {metadata.get('depth_range', ('未知', '未知'))}")