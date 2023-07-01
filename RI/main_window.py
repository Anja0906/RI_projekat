from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QPoint, QPointF


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.dragPosition = None
        self.dragging = False
        self.maximized = False
        self.dragPos = QPoint()
        self.set_up()

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.set_main_widget()

        # Sidebar widget and layout
        self.sidebar_widget = QWidget(self)
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.set_sidebar_widget()

        # Picture overview widget and layout
        self.image_label = QLabel('Odaberite sliku', self)
        self.set_picture_overview()

        # Resize and close buttons
        self.top_bar_layout = QHBoxLayout()
        self.close_button = QPushButton(self)
        self.minimize_button = QPushButton(self)
        self.maximize_button = QPushButton(self)
        self.set_close_button()

        # Window title
        self.content_label = QLabel('Program za citanje rukopisa', self)
        self.set_content_label()

        # Main part initialisations
        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.main_part_init()

        # Button add picture init
        self.button_add_picture = QPushButton("Dodaj sliku!")
        self.button_add_picture_init()

        # Button parse text init
        self.button_parse_text = QPushButton("Parsiraj tekst!")
        self.button_parse_text_init()

        # The text output part init
        self.textbox = QLineEdit(self)
        self.set_prompt()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.maximized:
                if (abs(event.position().x() - self.width()) < 10 or
                        abs(event.position().y() - self.height()) < 10):
                    self.dragging = True
                self.dragPos = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self.dragging and not self.maximized:
                self.resize(event.globalPosition().toPoint().x() - self.position().x(), event.globalPosition().toPoint().y() - self.position().y())
            elif not self.maximized:
                self.move((event.globalPosition() - QPointF(self.dragPos)).toPoint())
            event.accept()
        else:
            if (abs(event.position().x() - self.width()) < 10 or
                    abs(event.position().y() - self.height()) < 10):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            if (abs(event.position().x() - self.width()) < 10 or
                    abs(event.position().y() - self.height()) < 10):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def on_button_click(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(),
                                                     Qt.AspectRatioMode.KeepAspectRatio))
            # TODO: Ovde kod treba da pokupi sliku i da je sacuva

    def on_parse_click(self):
        self.textbox.setText("Tekst sa slike je: ksjdklsfjdslkfjds")
        # TODO: Ovde ide kod koji kupi sliku i parsira je

    def set_prompt(self):
        self.textbox.setStyleSheet("""
                    font: bold 16px;
                    background-color: #384061;
                    color: white;
                    border-style: none;
                    border-radius: 15px;
                    padding: 5px;
                """)
        self.textbox.setMinimumHeight(250)
        self.textbox.setDisabled(True)
        layout = QGridLayout()
        layout.addWidget(self.textbox, 0, 0)
        layout.setRowStretch(0, 1)
        self.content_layout.addLayout(layout, 4)

    def set_main_widget(self):
        self.main_widget.setStyleSheet("background-color: #24293E;")
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_widget.setLayout(self.main_layout)

    def set_sidebar_widget(self):
        self.sidebar_widget.setMinimumWidth(200)
        self.sidebar_widget.setStyleSheet("background-color: #384061;")
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)
        self.main_layout.addWidget(self.sidebar_widget)

    def set_picture_overview(self):
        self.image_label.setStyleSheet("""
                            font: bold 16px;
                            background-color: #384061;
                            color: white;
                            border-style: none;
                            border-radius: 15px;
                            padding: 5px;
                        """)
        self.image_label.setMinimumHeight(250)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QGridLayout()
        layout.addWidget(self.image_label, 0, 0)
        layout.setRowStretch(0, 1)

    def set_close_button(self):
        self.top_bar_layout.addStretch(1)
        self.close_button.setStyleSheet("border-style: none;")
        self.close_button.setIcon(QIcon('close_button.png'))  # Set your icon file here
        self.close_button.clicked.connect(self.close)
        self.close_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.minimize_button.setIcon(QIcon('minimize.png'))
        self.minimize_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.minimize_button.setStyleSheet("border-style: none;")
        self.maximize_button.setIcon(QIcon('maximize.png'))
        self.maximize_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.maximize_button.setStyleSheet("border-style: none;")
        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.maximize_restore)
        self.top_bar_layout.addWidget(self.minimize_button)
        self.top_bar_layout.addWidget(self.maximize_button)
        self.top_bar_layout.addWidget(self.close_button)

    def set_content_label(self):
        self.content_label.setStyleSheet("font: bold 25px; color: white")
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def button_add_picture_init(self):
        self.button_add_picture.setStyleSheet("""
                            font: bold 16px; 
                            background-color: #E68A00; 
                            color: white; 
                            border-radius: 20px;
                            margin: 10px;
                        """)
        self.button_add_picture.setFixedHeight(60)
        self.button_add_picture.setFixedWidth(180)
        self.button_add_picture.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.button_add_picture.clicked.connect(self.on_button_click)
        self.sidebar_layout.addWidget(self.button_add_picture)

    def button_parse_text_init(self):
        self.button_parse_text.setStyleSheet("""
                                    font: bold 16px; 
                                    background-color: #E68A00; 
                                    color: white; 
                                    border-radius: 20px;
                                    margin: 10px;
                                """)
        self.button_parse_text.setFixedHeight(60)
        self.button_parse_text.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_parse_text.setFixedWidth(180)
        self.button_parse_text.clicked.connect(self.on_parse_click)
        self.sidebar_layout.addWidget(self.button_parse_text)

    def main_part_init(self):
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.addLayout(self.top_bar_layout, 1)
        self.content_layout.addWidget(self.content_label, 1)
        self.main_layout.addWidget(self.sidebar_widget, 2)
        self.main_layout.addWidget(self.content_widget, 8)
        self.content_layout.addWidget(self.image_label, 4)

    def set_up(self):
        self.setWindowTitle("Program za citanje rukopisa")
        self.setGeometry(100, 100, 1000, 850)
        self.setStyleSheet("background-color: white;")
        self.setWindowFlag(Qt.WindowType.CustomizeWindowHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)


    def maximize_restore(self):
        if not self.maximized:
            self.showMaximized()
            self.maximized = True
        else:
            self.showNormal()
            self.maximized = False
