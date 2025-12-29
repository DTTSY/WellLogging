import sys
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
# from src.ui.mainwindow import MainWindow
from src.ui.mainwindow_uic import MainWindow

def main(argv):
    app = QApplication(argv)

    # Apply a built-in theme
    apply_stylesheet(app, theme='light_blue.xml')
    window = MainWindow(app)
    window.show()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)