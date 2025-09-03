import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        with open('stylesheet.qss', 'r') as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Warning: stylesheet.qss not found. Using default style.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
