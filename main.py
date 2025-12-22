import sys
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
# from src.ui.mainwindow import MainWindow
from src.ui.mainwindow_uic import MainWindow

app = QApplication(sys.argv)

# Apply a built-in theme
apply_stylesheet(app, theme='light_blue.xml')
window = MainWindow(app)
window.show()

app.exec()