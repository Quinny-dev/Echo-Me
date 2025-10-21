from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from pathlib import Path

READY_FILE = Path("gui_ready.flag")  # splash will wait for this file

class EchoMeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECHO ME")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))

        # ---- Fixed window setup ----
        self.setFixedSize(658, 780)
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowTitleHint |
            Qt.WindowCloseButtonHint |
            Qt.MSWindowsFixedSizeDialogHint
        )
        self.setStyleSheet("background-color: #333333;")
        self.center_top()

        # ---- Layout setup ----
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # ---- Top bar ----
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

        user_button = QPushButton("User")
        user_button.setFixedSize(80, 30)
        top_layout.addWidget(user_button)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        top_bar.setGraphicsEffect(shadow)
        main_layout.addWidget(top_bar)

        # ---- Camera area ----
        camera_frame = QFrame()
        camera_frame.setStyleSheet("background-color: #d3d3d3; border-radius: 10px;")
        camera_frame.setFixedHeight(450)
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")
        camera_layout.addWidget(self.camera_label)

        main_layout.addWidget(camera_frame)

        # ---- Menu panel  ----
        menu_frame = QFrame()
        menu_frame.setStyleSheet("border-radius: 10px;")
        menu_frame.setFixedHeight(60)
        menu_layout = QHBoxLayout(menu_frame)
        menu_layout.setContentsMargins(10, 10, 10, 10)
        menu_layout.setSpacing(20)

        # Button 1: Toggle Hand Overlay
        self.landmark_toggle_btn = QPushButton("Toggle Hand Overlay")
        self.landmark_toggle_btn.setCheckable(True)
        self.landmark_toggle_btn.setChecked(True)
        self.landmark_toggle_btn.setFixedSize(160, 30)
        self.landmark_toggle_btn.setStyleSheet(
        "background-color: #00b3b3; color: white; border-radius: 5px;"
        )
        self.landmark_toggle_btn.clicked.connect(self.toggle_landmarks)
        menu_layout.addWidget(self.landmark_toggle_btn, alignment=Qt.AlignLeft)

        # Button 2: Text to Speech
        self.text_to_speech_btn = QPushButton("Text to Speech")
        self.text_to_speech_btn.setCheckable(True)
        self.text_to_speech_btn.setChecked(True)
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.text_to_speech_btn.setStyleSheet(
            "background-color: #00b3b3; color: white; border-radius: 5px;"
        )
        menu_layout.addWidget(self.text_to_speech_btn, alignment=Qt.AlignCenter)

        # Button 3: Speech to Text
        self.speech_to_text_btn = QPushButton("Speech to Text")
        self.speech_to_text_btn.setCheckable(True)
        self.speech_to_text_btn.setChecked(True)
        self.speech_to_text_btn.setFixedSize(160, 30)
        self.speech_to_text_btn.setStyleSheet(
            "background-color: #00b3b3; color: white; border-radius: 5px;"
        )
        menu_layout.addWidget(self.speech_to_text_btn, alignment=Qt.AlignRight)

        main_layout.addWidget(menu_frame)

        # ---- Transcription panel ----
        transcription_label = QLabel("Transcription")
        transcription_label.setStyleSheet("background-color: #008080; color: white; padding: 5px;")
        transcription_label.setAlignment(Qt.AlignCenter)
        transcription_label.setFixedHeight(40)
        main_layout.addWidget(transcription_label)

        scroll_area = QScrollArea()
        self.transcription_content = QTextEdit()
        self.transcription_content.setReadOnly(True)
        self.transcription_content.setText("Transcription goes here...")
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.transcription_content)
        main_layout.addWidget(scroll_area)

        # ---- Hand Detector & Camera ----
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        # ---- Signal splash that GUI is ready ----
        self.signal_ready()

    def signal_ready(self):
        """Write ready file so splash can close 1s later."""
        try:
            READY_FILE.write_text("ready")
        except Exception:
            pass

    def toggle_landmarks(self):
        enabled = self.landmark_toggle_btn.isChecked()
        if self.camera:
            self.camera.set_draw_landmarks(enabled)
        print(f"Hand landmark overlay {'enabled' if enabled else 'disabled'}")

    def process_frame(self, frame, landmarks):
        pass

    def center_top(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = 0
        self.move(x, y)

    def moveEvent(self, event):
        self.center_top()

    def closeEvent(self, event):
        if self.camera:
            self.camera.release()
        if self.hand_detector:
            self.hand_detector.close()
        if READY_FILE.exists():
            READY_FILE.unlink()  # cleanup ready file
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = EchoMeApp()
    window.show()
    app.exec()