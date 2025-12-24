from PySide6.QtCore import QSize, Qt, QObject, Signal, QThread,QEvent,QThreadPool
from PySide6.QtWidgets import QMainWindow,QFileDialog,QMessageBox,QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,QWidget, QProgressDialog , QVBoxLayout
import pandas as pd
import os
from collections import defaultdict
from time import perf_counter
import traceback

from src.ui.ChartArea_muilt import MultiTrackWidget
from src.ui.DrillingPressureCalculator_window_ui import Ui_w_DrillingPressureCalculator
from src.ui.mainwindow_ui import Ui_APPMainWindow
from src.core.algorithm.DrillingPressureCalculator import DrillingPressureCalculator,OperationType
from src.models.wellModels import Well
from src.core.algorithm.CalculateForHomorock import HomorockCalculationThread, HomorockTask
from PySide6.QtCore import QTimer

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
    submitted = Signal()
    def __init__(self, parent=None,mw=None):
        super(DrillingPressureCalculator_ui, self).__init__(parent)
        self.setupUi(self)
        self.mw = mw
        # self.set
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
        self.submitted.emit(self.model['result'])
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
        fname, _ = QFileDialog.getOpenFileName(self, "Open main data file", "", "Table Files (*.csv,*.xls,*.xlsx,*.parquet);;Text Files (*.txt);;All Files (*)")
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


class MainWindow(QMainWindow, Ui_APPMainWindow):
    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.app = app #declare an app member
        self.setMinimumSize(QSize(1200, 800))
        self.dlg = DrillingPressureCalculator_ui()
        self.dlg.submitted.connect(self.set_mt)
        self.timer: QTimer = None

        # self.homorockCalculationThread = HomorockCalculationThread(self.well.static_data)

        self.mt = None
        # self.dataModel = dict()
        self.well = Well()

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
            fname, _ = QFileDialog.getOpenFileName(self, "Open data file", "", "Table Files (*.csv *.xls *.xlsx *.parquet);;Text Files (*.txt);;All Files (*)")
            if not fname:
                return
            try:
                # data = np.loadtxt(fname, delimiter=',')
                # metaInfo = dict()
                self.well.header['well_name'] = os.path.basename(fname).split('.')[0]
                data =  self.well.read_tableFile(fname)
                self.well.header['depth_range'] = (float(data.iloc[:,0].min()), float(data.iloc[:,0].max()))
                self.statusBar().showMessage(f"Loaded file: {os.path.basename(fname)}", 5000)
                # self.dataModel['log_data'] = data
                self.well.set_static_data(data)
                # self.well.header = metaInfo
                # self.dataModel['log_metaInfo'] = metaInfo
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
            self.set_mt(self.well.static_data)
        self.action_import_Ddata.triggered.connect(open_file)
        # open_action.triggered.connect(open_file)


        # edit_menu =menu_bar.addMenu("稳定性分析")
        self.action_subp1.triggered.connect(lambda: QMessageBox.warning(self, "地质力学静态计算", "方法未实现"))
        self.action_subp2.triggered.connect(self.open_drilling_pressure_calculator)
        self.action_subp3.triggered.connect(lambda: QMessageBox.warning(self, "机械扰动动态计算", "方法未实现"))
       

        # action1 = QAction("调整井深", self)
        self.pb_adjDepth.clicked.connect(self.adjustDepthValue)
        self.pb_homorock.clicked.connect(lambda: self.runHomorockCalculationThread(HomorockTask.BASE))
        self.pb_added_homorock.clicked.connect(lambda: self.runHomorockCalculationThread(HomorockTask.ADDED))
        self.pb_drilling_velocity.clicked.connect(lambda: self.runHomorockCalculationThread(HomorockTask.DRILLING_VELOCITY))


    def runHomorockCalculationThread(self, task: HomorockTask):
        if self.well.static_data is None:
            QMessageBox.warning(self, "数据缺失", "未导入有效数据")
            return
        if self.timer is not None:
            QMessageBox.information(self, "计算进行中", "请等待当前计算完成。")
            return
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        start_time = perf_counter()
        def _update_status():
            elapsed = perf_counter() - start_time
            self.statusBar().showMessage(f"{task} 计算中... 耗时: {elapsed:.1f} 秒")
        def _handle_homorock_result(df: pd.DataFrame, elapsed: float):
            try:
                # self.set_mt(df)
                # for i, t in enumerate(df.columns):
                #     w=300
                #     chart = self.mt.add_track(t, width=w, show_y_axis=(i == 0), x_range=(0, df[t].max()))
                #     # sample data: a shifted sine + noise per track
                #     x = df[t].to_numpy()
                #     chart.set_data(x, self.well.static_data.iloc[:, 0].to_numpy())
                # # self.mt.add_track(title=f"{task} 计算结果")
                # 将self.well.static_data的深度列与df拼接
                df.insert(0, self.well.static_data.columns[0], self.well.static_data.iloc[:, 0])
                tracks = []
                if task == HomorockTask.ADDED:
                    tracks = [['CollapsePressure_MPa_Original','CollapsePressure_MPa_Added'],['CollapsePressure_gcm3_Original','CollapsePressure_gcm3_Added'],'CollapsePressure_MPa_Increment','CollapsePressure_gcm3_Increment']
                self.mt.plot_dataframe(df, depth_column=self.well.static_data.columns[0], track_specs=tracks)
                self.timer.stop()
                self.timer.deleteLater()
                self.timer = None
                QMessageBox.information(self, "计算完成", f"{task}: 计算完成，耗时: {elapsed:.2f} 秒")
            except Exception as e:
                self.timer.stop()
                self.timer.deleteLater()
                self.timer = None
                QMessageBox.warning(self, "Error", f"Failed to set data:\n{e}")

        def _handle_homorock_error(message: str):
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
            self.statusBar().showMessage(f"计算失败，已停止。", 10000)
            QMessageBox.critical(self, f"{task}计算失败", f"计算失败:\n{message}")

        self.timer.timeout.connect(_update_status)
        self.timer.start()
        self.thread = HomorockCalculationThread(self.well.static_data, task)
        self.thread.result_ready.connect(_handle_homorock_result)
        self.thread.error_occurred.connect(_handle_homorock_error)
        self.thread.start()

    def HomorockCalculation(self, task):
        if self.well.static_data is None:
            QMessageBox.warning(self, "数据缺失", "未导入有效数据")
            return

        start_time = perf_counter()

        # Timer to update status bar with elapsed time
        timer = QTimer(self)
        timer.setInterval(500)
        def _update_status():
            elapsed = perf_counter() - start_time
            self.statusBar().showMessage(f"同质岩计算中... 耗时: {elapsed:.1f} 秒")
        timer.timeout.connect(_update_status)

        class _Worker(QObject):
            finished = Signal(pd.DataFrame)
            error = Signal(str)
            getResult = Signal()

            def __init__(self, task, static_data):
                super().__init__()
                self.task = task
                self.static_data = static_data

            def run(self):
                try:
                    if isinstance(self.task, HomorockCalculationThread):
                        t = self.task
                    else:
                        t = HomorockCalculationThread(self.static_data, self.task)

                    if hasattr(t, "run"):
                        try:
                            t.run()
                        except TypeError:
                            df = pd.DataFrame()
                            raise RuntimeError("HomorockCalculationThread run() method has incorrect signature")
                    else:
                        raise RuntimeError("HomorockCalculationThread has no run() method")

                    self.finished.emit(df)
                except Exception:
                    self.error.emit(traceback.format_exc())

        def _handle_homorock_finished(df: pd.DataFrame, thread: QThread, worker: QObject, start_time_local: float):
            try:
                if hasattr(self, '_homorock_timer') and self._homorock_timer:
                    try:
                        self._homorock_timer.stop()
                        self._homorock_timer.deleteLater()
                    except Exception:
                        pass
                    self._homorock_timer = None

                elapsed = perf_counter() - start_time_local
                self.statusBar().showMessage(f"同质岩计算完成，耗时: {elapsed:.2f} 秒", 10000)

                try:
                    self.set_mt(df)
                except Exception:
                    try:
                        self.well.set_static_data(df)
                        self.set_mt(self.well.static_data)
                    except Exception:
                        pass

                QMessageBox.information(self, "计算完成", "同质岩计算完成并已绘图。")
            finally:
                try:
                    worker.deleteLater()
                except Exception:
                    pass
                try:
                    thread.quit()
                except Exception:
                    pass
                self._homorock_thread = None
                self._homorock_worker = None

        def _handle_homorock_error(message: str, thread: QThread, worker: QObject, start_time_local: float):
            if hasattr(self, '_homorock_timer') and self._homorock_timer:
                try:
                    self._homorock_timer.stop()
                    self._homorock_timer.deleteLater()
                except Exception:
                    pass
                self._homorock_timer = None
            elapsed = perf_counter() - start_time_local
            self.statusBar().showMessage(f"同质岩计算失败，已停止。耗时: {elapsed:.2f} 秒", 10000)
            QMessageBox.critical(self, "计算失败", f"同质岩计算失败:\n{message}")
            try:
                worker.deleteLater()
            except Exception:
                pass
            try:
                thread.quit()
            except Exception:
                pass
            self._homorock_thread = None
            self._homorock_worker = None

        thread = QThread(self)
        worker = _Worker(task, self.well.static_data)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(lambda df: _handle_homorock_finished(df, thread, worker, start_time))
        worker.error.connect(lambda msg: _handle_homorock_error(msg, thread, worker, start_time))

        thread.finished.connect(thread.deleteLater)

        # keep refs for potential cancellation/inspection
        self._homorock_thread = thread
        self._homorock_worker = worker
        self._homorock_timer = timer

        timer.start()
        thread.start()

    def open_drilling_pressure_calculator(self):
        self.dlg.show()

    def quit_app(self):
        self.app.quit()

    def toolbar_button_click(self):
        self.statusBar().showMessage("Message from my app",3000)
    
    def adjustDepthValue(self):
        # 打开一个对话课框，获取用户输入的深度值，最大值和最小值
        if self.well.static_data is None:
            QMessageBox.warning(self, "数据缺失", "未导入有效数据")
            return

        # determine sensible defaults

        self.dsp_startDepth.value()
        dmin = float(self.dsp_startDepth.value())
        dmax = float(self.dsp_endDepth.value())
        if dmin >= dmax:
            QMessageBox.warning(self, "深度范围错误", "起始深度需小于终止深度")
            return

        try:
            # self.set_mt(self.well.static_data, depth_range=(dmin, dmax))
            self.mt.set_depth_range(dmin, dmax)
            self.statusBar().showMessage(f"井名: {self.well.header.get('well_name', '未知')}  深度范围: {dmin} - {dmax}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply depth range:\n{e}")
        
    
    def set_mt(self, data:pd.DataFrame, depth_range=None):
        if self.mt is not None:
            self.mt.setParent(None)  # Remove existing widget
            self.mt.deleteLater()
            self.mt = None
        # data = self.dataModel['log_data']
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
        # fixed_gap = 2
        # mt._tracks_layout.setSpacing(fixed_gap)

        # Place the multi-track widget in the main window
        # self.setCentralWidget(mt)
        # if self.mt is None:
        #     self.gridLayout_main_left.replaceWidget(self.mt, mt)
        # else:
        self.mt = mt
        self.gridLayout_main_left.addWidget(self.mt)
        self.statusBar().showMessage(f"井名: {self.well.header.get('well_name', '未知')}  深度范围: {self.well.header.get('depth_range', ('未知', '未知'))}")
    
