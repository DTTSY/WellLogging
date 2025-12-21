from PySide6.QtCore import QSize, Qt, QObject, Signal, QThread,QEvent
from PySide6.QtWidgets import QMainWindow,QFileDialog,QMessageBox,QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,QWidget, QProgressDialog , QVBoxLayout
import pandas as pd
import os
from collections import defaultdict
from time import perf_counter
import traceback

from src.ui.ChartArea import MultiTrackWidget
from src.ui.DrillingPressureCalculator_window_ui import Ui_w_DrillingPressureCalculator
from src.ui.mainwindow_ui import Ui_APPMainWindow
from src.core.algorithm.DrillingPressureCalculator import DrillingPressureCalculator,OperationType


class CalculationWorker(QObject):
    progress = Signal(int, str, float)
    finished = Signal(pd.DataFrame, float)
    error = Signal(str)

    def __init__(self, main_data: pd.DataFrame, cf_data, operation: OperationType, calculator_params: dict):
        super().__init__()
        self._main_data = main_data
        self._cf_data = cf_data
        self._operation = operation
        self._calculator_params = calculator_params

    def _emit_progress(self, value: int, message: str, start_time: float):
        elapsed = perf_counter() - start_time
        self.progress.emit(value, message, elapsed)

    def run(self):
        start_time = perf_counter()
        try:
            calculator = DrillingPressureCalculator()
            for attr, val in self._calculator_params.items():
                setattr(calculator, attr, val)

            df = self._main_data.copy(deep=True)

            self._emit_progress(5, "准备计算参数", start_time)
            df = calculator.calculate_a_values(df)
            self._emit_progress(25, "已计算a值", start_time)
            df = calculator.calculate_v_values(df)
            self._emit_progress(45, "已计算v值", start_time)
            df = calculator.fill_collapse_fracture_data(df, self._cf_data)
            self._emit_progress(65, "已填充坍塌破裂数据", start_time)
            df = calculator.calculate_all_pressures(df, self._operation)
            self._emit_progress(85, "已计算钻井压力", start_time)
            calculator.save_results(df, self._operation)
            # self._emit_progress(100, "已保存结果", start_time)

            total_elapsed = perf_counter() - start_time
            self.finished.emit(df, total_elapsed)
        except Exception as exc:
            message = traceback.format_exc()
            self.error.emit(message)

class DrillingPressureCalculator_ui(QWidget, Ui_w_DrillingPressureCalculator):
    def __init__(self, parent=None,mw=None):
        super(DrillingPressureCalculator_ui, self).__init__(parent)
        self.setupUi(self)
        self.mw = mw
        # Additional initialization code can go here
        self.calculator = DrillingPressureCalculator()
        self.pb_open_main.clicked.connect(lambda : self.open_file('main_data'))
        self.pb_open_dpf.clicked.connect(lambda : self.open_file('cf_data'))
        self.model = defaultdict(lambda: None)
        self._calc_thread = None
        self._worker = None
        self._progress_dialog = None

        self.pb_up.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_OUT))
        self.pb_down.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_IN))

    def handle_move(self, operation: OperationType):
        if self.data_validation():
            self.drilling_action(operation=operation)

    def data_validation(self):
        # TODO: validate input data from UI
        if self.model['main_data'] is None:
            QMessageBox.warning(self, "数据缺失", "请先导入主数据文件")
            return False
        if self.model['cf_data'] is None: 
            QMessageBox.warning(self, "数据缺失", "请先导入坍塌破裂压力数据文件")
            return False
        self.calculator.Dh = self.dsb_Dh.value()
        self.calculator.Dhi = self.dsb_Dhi.value()
        self.calculator.fai300 = self.dsb_fai300.value()
        self.calculator.fai600 = self.dsb_fai600.value()
        return True
    
    def drilling_action(self, operation: OperationType):
        if self._calc_thread is not None:
            QMessageBox.information(self, "计算进行中", "请等待当前计算完成。")
            return

        if self.model['main_data'] is None or self.model['cf_data'] is None:
            QMessageBox.warning(self, "数据缺失", "请先导入所需的数据文件。")
            return

        calculator_params = {
            'Dh': self.calculator.Dh,
            'Dhi': self.calculator.Dhi,
            'fai300': self.calculator.fai300,
            'fai600': self.calculator.fai600
        }

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
        worker = CalculationWorker(
            main_data=self.model['main_data'],
            cf_data=self.model['cf_data'],
            operation=operation,
            calculator_params=calculator_params
        )
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
        self.model['result'] = df
        if self._progress_dialog is not None:
            self._progress_dialog.setValue(100)
            self._progress_dialog.setLabelText(f"已保存结果\n总耗时: {elapsed:.2f} 秒")
            self._progress_dialog.close()
            self._progress_dialog = None
        self.pb_up.setEnabled(True)
        self.pb_down.setEnabled(True)
        QMessageBox.information(self, "计算完成", f"计算耗时: {elapsed:.2f} 秒，钻井压力计算已完成并保存结果文件。")

    def _handle_calculation_error(self, message: str):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self.pb_up.setEnabled(True)
        self.pb_down.setEnabled(True)
        QMessageBox.critical(self, "计算失败", message)

    def _on_calculation_thread_finished(self):
        self._calc_thread = None
        self._worker = None



    def open_file(self,file_type:str):
        fname, _ = QFileDialog.getOpenFileName(self, "Open main data file", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if not fname:
            QMessageBox.information(self, "No File Selected", "No file was selected.")
            return
        try:
            if file_type == 'main_data':
                self.model[file_type] = self.calculator.read_main_data(fname)
            elif file_type == 'cf_data':
                self.model[file_type] = self.calculator.read_collapse_fracture_data(fname)
            QMessageBox.information(self, "File Loaded", f"Successfully loaded file: {os.path.basename(fname)}")
            # You can now use 'data' with your DrillingPressureCalculator instance
        except Exception as e:
            QMessageBox.warning(self, "Load error", f"Failed to load file:\n{e}")
            return

class MainWindow_c(QMainWindow, Ui_APPMainWindow):
    def __init__(self, app):
        super(MainWindow_c, self).__init__()
        self.setupUi(self)
        self.app = app #declare an app member
        self.setMinimumSize(QSize(1200, 800))
        self.mt = None
        self.dataModel = dict()

        #Menubar and menus
        # menu_bar = self.menuBar()
        # file_menu = menu_bar.addMenu("文件")
        # # new_action =  file_menu.addAction("New")
        # open_action = file_menu.addAction("导入 地质力学数据")

        # open_d_action = file_menu.addAction("导入 钻井动态数据")

        self.action_import_DTdata.triggered.connect(lambda: QMessageBox.warning(self, "导入 钻井动态数据", "方法未实现"))
        # open_action.setShortcut("Ctrl+O")
        # open_action.setStatusTip("Open a CSV/text file (value, depth)")

        def open_file():
            fname, _ = QFileDialog.getOpenFileName(self, "Open data file", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
            if not fname:
                return
            try:
                # data = np.loadtxt(fname, delimiter=',')
                metaInfo = dict()
                metaInfo['well_name'] = os.path.basename(fname).split('.')[0]
                data =  pd.read_csv(fname)
                metaInfo['depth_range'] = (float(data.iloc[:,0].min()), float(data.iloc[:,0].max()))
                self.statusBar().showMessage(f"Loaded file: {os.path.basename(fname)}", 5000)
                self.dataModel['log_data'] = data
                self.dataModel['log_metaInfo'] = metaInfo
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
            self.set_mt(data)
        self.action_import_Ddata.triggered.connect(open_file)
        # open_action.triggered.connect(open_file)


        # edit_menu =menu_bar.addMenu("稳定性分析")
        self.action_subp1.triggered.connect(lambda: QMessageBox.warning(self, "地质力学静态计算", "方法未实现"))
        self.action_subp2.triggered.connect(self.open_drilling_pressure_calculator)
        self.action_subp3.triggered.connect(lambda: QMessageBox.warning(self, "机械扰动动态计算", "方法未实现"))
       

        # action1 = QAction("调整井深", self)

        self.pb_adjDepth.clicked.connect(self.adjustDepthValue)

        # image = QImage("assets/images/start.jpg")
        # self.w_ff.setCentralWidget(QLabel(pixmap=QPixmap.fromImage(image)))
        # self.gb_main_left_area.setCentralWidget(QLabel(pixmap=QPixmap.fromImage(image)))
        # 向group box中添加图片
        # image = QImage("assets/images/start.jpg")
        # self._left_image_pixmap = image
        # self.left_image_label = QLabel(self.gb_main_left_area)
        # self.left_image_label.setAlignment(Qt.AlignCenter)
        # left_area_layout = QVBoxLayout()
        # left_area_layout.setContentsMargins(0, 0, 0, 0)
        # left_area_layout.addWidget(self.left_image_label)
        # left_area_layout.addWidget(QLabel("欢迎使用井下数据可视化与分析系统",alignment=Qt.AlignmentFlag.AlignCenter))
        # self.gb_main_left_area.setLayout(left_area_layout)
        self.gridLayout_main_left.addWidget(QLabel("欢迎使用井下数据可视化与分析系统",alignment=Qt.AlignmentFlag.AlignCenter))
        # self._update_left_image_pixmap()
        # self.gb_main_left_area.installEventFilter(self)

    # def _update_left_image_pixmap(self):
    #     if hasattr(self, "left_image_label") and hasattr(self, "_left_image_pixmap"):
    #         target_size = self.gb_main_left_area.size()
    #         self.left_image_label.setPixmap(
    #             self._left_image_pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    #         )

    # def eventFilter(self, obj, event):
    #     if obj is self.gb_main_left_area and event.type() == QEvent.Resize:
    #         self._update_left_image_pixmap()
    #     return super().eventFilter(obj, event)

    def open_drilling_pressure_calculator(self):
        self.dlg = DrillingPressureCalculator_ui()
        self.dlg.show()

    def quit_app(self):
        self.app.quit()

    def toolbar_button_click(self):
        self.statusBar().showMessage("Message from my app",3000)
    
    def adjustDepthValue(self):
        # 打开一个对话课框，获取用户输入的深度值，最大值和最小值
        if 'log_data' not in self.dataModel:
            QMessageBox.warning(self, "数据缺失", "未导入有效数据")
            return

        # determine sensible defaults
        data = self.dataModel.get('log_data')
        try:
            default_min = float(data.iloc[:, 0].min())
            default_max = float(data.iloc[:, 0].max())
        except Exception:
            default_min, default_max = 0.0, 100.0

        self.dsp_startDepth.value()
        dmin = float(self.dsp_startDepth.value())
        dmax = float(self.dsp_endDepth.value())
        if dmin >= dmax:
            QMessageBox.warning(self, "深度范围错误", "起始深度需小于终止深度")
            return

        try:
            self.set_mt(self.dataModel.get('log_data'), depth_range=(dmin, dmax))
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply depth range:\n{e}")
        
    
    def set_mt(self, data:pd.DataFrame, depth_range=None):
        if self.mt is not None:
            self.mt.setParent(None)  # Remove existing widget
            self.mt.deleteLater()
            self.mt = None
        data = self.dataModel['log_data']
        if depth_range is not None:
            dmin, dmax = depth_range
        else:
            dmin = data.iloc[:, 0].min()
            dmax = data.iloc[:, 0].max()
        mt = MultiTrackWidget(depth_range=(dmin, dmax))

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
        # self.setCentralWidget(mt)
        # if self.mt is None:
        #     self.gridLayout_main_left.replaceWidget(self.mt, mt)
        # else:
        self.mt = mt
        self.gridLayout_main_left.addWidget(self.mt)
        self.statusBar().showMessage(f"井名: {self.dataModel.get('log_metaInfo', {}).get('well_name', '未知')}  深度范围: {self.dataModel.get('log_metaInfo', {}).get('depth_range', ('未知', '未知'))}")