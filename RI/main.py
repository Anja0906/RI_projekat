from PyQt6.QtWidgets import QApplication

import model
from main_window import MainWindow


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()





