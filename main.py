import sys
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
# from src.ui.mainwindow import MainWindow
from src.ui.mainwindow_uic import MainWindow_c

app = QApplication(sys.argv)

# Apply a built-in theme
apply_stylesheet(app, theme='light_blue.xml')
window = MainWindow_c(app)
window.show()

app.exec()