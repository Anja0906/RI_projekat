from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Program za citanje rukopisa")
        self.setGeometry(100, 100, 1000, 850)
        self.setStyleSheet("background-color: white;")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.main_layout.setSpacing(0)  # Remove spacing
        self.main_widget.setLayout(self.main_layout)

        self.sidebar_widget = QWidget(self)
        self.sidebar_widget.setMinimumWidth(100)
        self.sidebar_widget.setStyleSheet("background-color: #24293E;")  # This sets the background color of the sidebar
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)
        self.main_layout.addWidget(self.sidebar_widget)

        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.sidebar_widget, 2)
        self.main_layout.addWidget(self.content_widget, 8)
        self.button = QPushButton("Click Me!")
        self.button.setStyleSheet("""
                    font: bold 16px; 
                    background-color: #8EBBFF; 
                    color: white; 
                    border-radius: 20px;
                    margin: 10px;
                """)
        self.button.setFixedHeight(60)
        self.button.setCursor(QCursor(Qt.PointingHandCursor))
        self.sidebar_layout.addWidget(self.button)

        self.content_label = QLabel('Main content area', self)
        self.content_label.setStyleSheet("font: 25px; background-color: light gray; color: black")
        self.content_layout.addWidget(self.content_label)

        self.button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        self.content_label.setText(self.content_label.text() + "\nButton clicked!")
