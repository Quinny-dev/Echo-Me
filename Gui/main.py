from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QColor

class EchoMeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECHO ME")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: #333333;") 

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        

        #Top bar
        top_bar = QFrame()
        top_bar.setFixedHeight(60)
        top_bar.setStyleSheet("background-color: #008080;") 
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 0, 10, 0)

        logo_label = QLabel("ECHO ME")
        logo_label.setFont(QFont("Arial", 16, QFont.Bold))
        logo_label.setStyleSheet("color: white;")
        top_layout.addWidget(logo_label)

        top_layout.addStretch()

        user_button = QPushButton("👤")
        user_button.setFixedSize(40, 40)
        top_layout.addWidget(user_button)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)               # how soft the shadow is
        shadow.setXOffset(0)
        shadow.setYOffset(20)                   # slight downward shadow
        shadow.setColor(QColor(0, 0, 0, 120)) # semi-transparent black
        top_bar.setGraphicsEffect(shadow)

        main_layout.addWidget(top_bar)

        #Camera area
        camera_frame = QFrame()
        camera_frame.setStyleSheet("background-color: #d3d3d3; border-radius: 10px;")
        camera_frame.setFixedHeight(450)
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setAlignment(Qt.AlignCenter)
        camera_container = QVBoxLayout()
        camera_container.setContentsMargins(100, 10, 10, 100)
      
        main_layout.addLayout(camera_container)
        camera_icon = QLabel("📷")
        camera_icon.setFont(QFont("Arial", 48))
        camera_layout.addWidget(camera_icon)

        #Camera panel buttons like pinning and expanding fullscreen
        camera_buttons_layout = QHBoxLayout()
        pin_button = QPushButton("📌")
        pin_button.setFixedSize(30, 30)
        camera_buttons_layout.addWidget(pin_button, alignment=Qt.AlignLeft)

        fullscreen_button = QPushButton("⤢")
        fullscreen_button.setFixedSize(30, 30)
        camera_buttons_layout.addWidget(fullscreen_button, alignment=Qt.AlignRight)

        camera_layout.addLayout(camera_buttons_layout)
        main_layout.addWidget(camera_frame)

        #Transcription panel for the transcription....to transcribe...
        transcription_label = QLabel("Transcription")
        transcription_label.setStyleSheet("background-color: #008080; color: white; padding: 5px;")
        transcription_label.setAlignment(Qt.AlignCenter)
        transcription_label.setFixedHeight(80)
        main_layout.addWidget(transcription_label)
        scroll_area = QScrollArea()
        transcription_container = QVBoxLayout()
        transcription_container.setContentsMargins(10, 10, 10, 10)
        transcription_container.addWidget(scroll_area)
        main_layout.addLayout(transcription_container)
        scroll_area.setWidgetResizable(True)
        transcription_content = QTextEdit()
        transcription_content.setReadOnly(True)
        transcription_content.setText("Transcription goes here...")
        scroll_area.setWidget(transcription_content)

        main_layout.addWidget(scroll_area)


if __name__ == "__main__":
    app = QApplication([])
    window = EchoMeApp()
    window.show()
    app.exec()
