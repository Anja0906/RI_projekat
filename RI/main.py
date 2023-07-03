from PyQt6.QtWidgets import QApplication

import model
from main_window import MainWindow


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
    model.predict_image("/Users/bane/Downloads/WhatsApp Image 2023-07-02 at 19.41.39.jpeg")
    # model.boxing_model_training()





