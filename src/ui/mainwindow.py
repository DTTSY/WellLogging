from PySide6.QtCore import QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow,QToolBar,QStatusBar,QFileDialog,QMessageBox,QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,QWidget
import pandas as pd
import os
from collections import defaultdict

from src.ui.ChartArea import MultiTrackWidget
from src.ui.DrillingPressureCalculator_window_ui import Ui_w_DrillingPressureCalculator
from src.core.algorithm.DrillingPressureCalculator import DrillingPressureCalculator,OperationType

class DrillingPressureCalculator_ui(QWidget, Ui_w_DrillingPressureCalculator):
    def __init__(self, parent=None,mw=None):
        super(DrillingPressureCalculator_ui, self).__init__(parent)
        self.setupUi(self)
        self.mw = mw
        # Additional initialization code can go here
        self.calculator = DrillingPressureCalculator()
        self.pb_open_main.clicked.connect(lambda : self.open_file('main_data'))
        self.pb_open_dpf.clicked.connect(lambda : self.open_file('cf_data'))
        self.model = defaultdict(int)

        self.pb_up.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_OUT))
        self.pb_down.clicked.connect(lambda: self.handle_move(OperationType.TRIPPING_IN))

    def handle_move(self, operation: OperationType):
        self.data_validation()
        self.drilling_action(operation=operation)

    def data_validation(self):
        # TODO: validate input data from UI
        if self.model['main_data'] is None:
            QMessageBox.warning(self, "数据缺失", "请先导入主数据文件")
            return
        if self.model['cf_data'] is None: 
            QMessageBox.warning(self, "数据缺失", "请先导入坍塌破裂压力数据文件")
            return
        self.calculator.Dh = self.dsb_Dh.value()
        self.calculator.Dhi = self.dsb_Dhi.value()
        self.calculator.fai300 = self.dsb_fai300.value()
        self.calculator.fai600 = self.dsb_fai600.value()

    def drilling_action(self, operation: OperationType):
        # TODO 弹出等待对话框
        
        # 8. 计算a值和v值
        df = self.calculator.calculate_a_values(self.model['main_data'])
        df = self.calculator.calculate_v_values(df)
        # 9. 填充坍塌破裂压力数据
        df = self.calculator.fill_collapse_fracture_data(df, self.model['cf_data'])
        # 10. 根据操作类型计算压力
        df = self.calculator.calculate_all_pressures(df, operation)
        # 11. 保存结果
        self.model['result'] = df
        self.calculator.save_results(df, operation)
        QMessageBox.information(self, "计算完成", "钻井压力计算已完成并保存结果文件。")



    def open_file(self,file_type:str):
        fname, _ = QFileDialog.getOpenFileName(self, "Open main data file", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)")
        if not fname:
            QMessageBox.information(self, "No File Selected", "No file was selected.")
            return
        try:
            if file_type == 'main_data':
                self.model[file_type] = self.calculator.read_main_data(fname)
            elif file_type == 'dp_data':
                self.model[file_type] = self.calculator.read_collapse_fracture_data(fname)
            QMessageBox.information(self, "File Loaded", f"Successfully loaded file: {os.path.basename(fname)}")
            # You can now use 'data' with your DrillingPressureCalculator instance
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
        self.dataModel = dict()

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
        self.statusBar().showMessage(f"井名: {self.dataModel.get('log_metaInfo', {}).get('well_name', '未知')}  深度范围: {self.dataModel.get('log_metaInfo', {}).get('depth_range', ('未知', '未知'))}")