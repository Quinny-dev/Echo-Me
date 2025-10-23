from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QDialog, QFormLayout, QCheckBox, QComboBox
)
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtCore import Qt
from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from pathlib import Path

READY_FILE = Path("gui_ready.flag")  # splash will wait for this file

# ---- User Preferences Dialog ----
class PreferencesDialog(QDialog):
    def __init__(self, parent=None, show_landmarks=True, dark_mode=True, translation="No Translation", voice="English (US)", speed="Normal"):
        super().__init__(parent)
        self.setWindowTitle("User Preferences")
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(350, 300)

        self.parent_window = parent

        self.dark_mode_checkbox = QCheckBox("Enable Dark Mode")
        self.dark_mode_checkbox.setChecked(dark_mode)

        self.show_landmarks_checkbox = QCheckBox("Show landmark overlay")
        self.show_landmarks_checkbox.setChecked(show_landmarks)

        # Translation dropdown
        self.translation_combo = QComboBox()
        self.translation_combo.addItems([
            "No Translation", "English (US)", "English (UK)", "Afrikaans", "Spanish (Spain)", "French (France)"
        ])
        self.translation_combo.setCurrentText(translation)

        # Voice dropdown
        self.voice_combo = QComboBox()
        self.voice_combo.addItems([
            "English (US)", "English (UK)", "Afrikaans", "Spanish (Spain)", "French (France)"
        ])
        self.voice_combo.setCurrentText(voice)

        # Speed dropdown
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Slow", "Normal", "Fast"])
        self.speed_combo.setCurrentText(speed)

        # Save button
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_preferences)

        layout = QFormLayout(self)
        layout.addRow(self.dark_mode_checkbox)
        layout.addRow(self.show_landmarks_checkbox)
        layout.addRow(QLabel("Translate to:"), self.translation_combo)
        layout.addRow(QLabel("Voice:"), self.voice_combo)
        layout.addRow(QLabel("Speed:"), self.speed_combo)
        layout.addRow(save_btn)

        # Apply styles based on dark mode
        self.apply_styles(dark_mode)

    def apply_styles(self, dark_mode):
        if dark_mode:
            bg_color = "#222"
            text_color = "white"
            checkbox_border = "#008080"
            checkbox_checked = "#339999"
            button_color = "#008080"
            button_hover = "#00b3b3"
            combobox_bg = "#333333"
            combobox_text = "white"
        else:
            bg_color = "white"
            text_color = "black"
            checkbox_border = "#008080"
            checkbox_checked = "#339999"
            button_color = "#008080"
            button_hover = "#00b3b3"
            combobox_bg = "white"
            combobox_text = "black"

        self.setStyleSheet(f"""
            QDialog {{ background-color: {bg_color}; color: {text_color}; border-radius: 10px; }}
            QLabel {{ color: {text_color}; }}
            QCheckBox {{ spacing: 6px; color: {text_color}; }}
            QCheckBox::indicator {{
                width: 18px; height: 18px; border-radius: 4px;
                border: 2px solid {checkbox_border}; background-color: transparent;
            }}
            QCheckBox::indicator:checked {{
                background-color: {checkbox_checked}; border: 2px solid {checkbox_checked};
            }}
            QPushButton {{
                background-color: {button_color}; color: {text_color}; border-radius: 5px; height: 30px;
            }}
            QPushButton:hover {{
                background-color: {button_hover};
            }}
            QComboBox {{
                background-color: {combobox_bg};
                color: {combobox_text};
                border-radius: 5px;
                padding: 2px 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {combobox_bg};
                color: {combobox_text};
                selection-background-color: #339999;
                selection-color: {combobox_text};
            }}
        """)

    def save_preferences(self):
        if self.parent_window:
            self.parent_window.landmark_toggle_btn_state(self.show_landmarks_checkbox.isChecked())
            self.parent_window.set_dark_mode(self.dark_mode_checkbox.isChecked())
            # Update TTS preferences
            self.parent_window.tts_translation = self.translation_combo.currentText()
            self.parent_window.tts_voice = self.voice_combo.currentText()
            self.parent_window.tts_speed = self.speed_combo.currentText()
        self.accept()


class EchoMeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECHO ME")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))

        self.setFixedSize(658, 780)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.dark_mode = True
        self.center_top()

        # TTS preferences
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"

        # ---- Layout setup ----
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        # ---- Top bar ----
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(60)
        self.top_layout = QHBoxLayout(self.top_bar)
        self.top_layout.setContentsMargins(10, 0, 10, 0)

        self.logo_label = QLabel("ECHO ME")
        self.logo_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.top_layout.addWidget(self.logo_label)
        self.top_layout.addStretch()

        self.user_button = QPushButton("User")
        self.user_button.setFixedSize(80, 30)
        self.user_button.clicked.connect(self.open_preferences)
        self.top_layout.addWidget(self.user_button)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.top_bar.setGraphicsEffect(shadow)
        self.main_layout.addWidget(self.top_bar)

        # ---- Camera area ----
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        self.camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_layout.addWidget(self.camera_label)
        self.main_layout.addWidget(self.camera_frame)

        # ---- Menu panel ----
        self.menu_frame = QFrame()
        self.menu_frame.setFixedHeight(60)
        self.menu_layout = QHBoxLayout(self.menu_frame)
        self.menu_layout.setContentsMargins(10, 10, 10, 10)
        self.menu_layout.setSpacing(20)

        self.text_to_speech_btn = QPushButton("Text to Speech")
        self.text_to_speech_btn.setCheckable(True)
        self.text_to_speech_btn.setChecked(True)
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.menu_layout.addWidget(self.text_to_speech_btn, alignment=Qt.AlignLeft)

        self.speech_to_text_btn = QPushButton("Speech to Text")
        self.speech_to_text_btn.setCheckable(True)
        self.speech_to_text_btn.setChecked(True)
        self.speech_to_text_btn.setFixedSize(160, 30)
        self.menu_layout.addWidget(self.speech_to_text_btn, alignment=Qt.AlignRight)

        self.main_layout.addWidget(self.menu_frame)

        # ---- Transcription panel ----
        self.transcription_label = QLabel("Transcription")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setFixedHeight(40)
        self.main_layout.addWidget(self.transcription_label)

        self.scroll_area = QScrollArea()
        self.transcription_content = QTextEdit()
        self.transcription_content.setReadOnly(True)
        self.transcription_content.setText("Transcription goes here...")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.transcription_content)
        self.main_layout.addWidget(self.scroll_area)

        # ---- Hand Detector & Camera ----
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        # ---- Landmark overlay state ----
        self.show_landmarks = True

        # ---- Signal splash that GUI is ready ----
        self.signal_ready()

        # Apply initial dark-mode
        self.set_dark_mode(self.dark_mode)

    def set_dark_mode(self, dark: bool):
        self.dark_mode = dark
        if dark:
            bg_color = "#333333"
            top_bar_color = "#008080"
            menu_btn_color = "#00b3b3"
            hover_color = "#66ffff"
            transcription_bg = "#008080"
            text_color = "white"
        else:
            bg_color = "#f5f5f5"
            top_bar_color = "#00b3b3"
            menu_btn_color = "#008080"
            hover_color = "#00cccc"
            transcription_bg = "#00b3b3"
            text_color = "black"

        # Apply styles
        self.setStyleSheet(f"background-color: {bg_color};")
        self.top_bar.setStyleSheet(f"background-color: {top_bar_color};")
        self.logo_label.setStyleSheet(f"color: {text_color};")
        for btn in [self.user_button, self.text_to_speech_btn, self.speech_to_text_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {menu_btn_color};
                    color: {text_color};
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
            """)
        self.camera_frame.setStyleSheet(f"background-color: {bg_color}; border-radius: 10px;")
        self.transcription_label.setStyleSheet(f"background-color: {transcription_bg}; color: {text_color}; padding: 5px;")
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")

    def open_preferences(self):
        dialog = PreferencesDialog(
            self,
            show_landmarks=self.show_landmarks,
            dark_mode=self.dark_mode,
            translation=getattr(self, "tts_translation", "No Translation"),
            voice=getattr(self, "tts_voice", "English (US)"),
            speed=getattr(self, "tts_speed", "Normal")
        )
        dialog.exec()

    def landmark_toggle_btn_state(self, state):
        self.show_landmarks = state
        if self.camera:
            self.camera.set_draw_landmarks(state)
        print(f"Hand landmark overlay {'enabled' if state else 'disabled'}")

    def signal_ready(self):
        try:
            READY_FILE.write_text("ready")
        except Exception:
            pass

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
            READY_FILE.unlink()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = EchoMeApp()
    window.show()
    app.exec()