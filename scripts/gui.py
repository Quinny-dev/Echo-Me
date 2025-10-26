"""
Main GUI Module - Display Only
Handles the main window and UI layout, delegates functionality to specialized modules
"""

from PySide6.QtWidgets import ( 
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QTabWidget, QStyle
)
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtCore import Qt

# Import external modules
import tensorflow as tf
import numpy as np
from model_handler import ModelHandler
from user_data import get_user_preferences
from tts_handler import TTSHandler
from stt_handler import STTHandler
from preferences_dialog import PreferencesDialog
from styling import ThemeManager, apply_theme
from camera_handler import CameraHandler
from model_handler import ModelHandler
from login import show_login_flow


class EchoMeApp(QWidget):
    """Main application window - handles display and UI coordination"""
    
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.theme_manager = ThemeManager()
        self.dark_mode = True  # default until preferences load
        
        self.tts_handler = None
        self.stt_handler = None
        self.camera_handler = None
        self.model_handler = None
        
        self.setup_window()
        self.setup_ui()
        self.setup_handlers()
        self.load_user_preferences()
        self.apply_styling()

        # Connect model predictions to update the transcription box
        self.model_handler.prediction_made.connect(self.on_model_prediction)

        print("✅ EchoMeApp Initialized and Ready")

    # ------------------- Window Setup -------------------
    def setup_window(self):
        self.setWindowTitle(f"ECHO ME - {self.username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))
        self.setFixedSize(658, 780)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.center_window()
    
    def center_window(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = 50
        self.move(x, y)
    
    # ------------------- UI Setup -------------------
    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)
        
        self.create_top_bar()
        self.create_camera_area()
        self.create_tabbed_interface()
    
    def create_top_bar(self):
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(80)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(10, 0, 10, 0)

        # Logo
        self.logo_label = QLabel("ECHO ME")
        self.logo_label.setFont(QFont("Arial", 16, QFont.Bold))
        top_layout.addWidget(self.logo_label)
        top_layout.addStretch()

        # User button
        self.user_button = QPushButton(self.username)
        self.user_button.setFixedSize(80, 30)
        self.user_button.clicked.connect(self.open_preferences)
        top_layout.addWidget(self.user_button)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.top_bar.setGraphicsEffect(shadow)

        self.main_layout.addWidget(self.top_bar)
    
    def create_camera_area(self):
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        camera_layout = QVBoxLayout(self.camera_frame)
        camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        camera_layout.addWidget(self.camera_label)

        self.main_layout.addWidget(self.camera_frame)
    
    def create_tabbed_interface(self):
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        self.create_tts_tab()
        self.create_stt_tab()
    
    def create_tts_tab(self):
        self.tts_tab = QWidget()
        layout = QVBoxLayout(self.tts_tab)

        # Buttons Row
        btn_layout = QHBoxLayout()
        self.text_to_speech_btn = QPushButton("🔊 Text to Speech")
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.text_to_speech_btn.clicked.connect(self.handle_text_to_speech)
        btn_layout.addWidget(self.text_to_speech_btn)

        self.download_audio_btn = QPushButton("📥 Download Audio")
        self.download_audio_btn.setFixedSize(160, 30)
        self.download_audio_btn.setEnabled(False)
        self.download_audio_btn.clicked.connect(self.handle_download_audio)
        btn_layout.addWidget(self.download_audio_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Text Area
        self.tts_scroll = QScrollArea()
        self.tts_content = QTextEdit()
        self.tts_content.setText("Enter text here for Text-To-Speech...")
        self.tts_content.focusInEvent = self.clear_placeholder_tts
        self.tts_scroll.setWidgetResizable(True)
        self.tts_scroll.setWidget(self.tts_content)
        layout.addWidget(self.tts_scroll)

        self.tabs.addTab(self.tts_tab, "Text To Speech")
    
    def create_stt_tab(self):
        self.stt_tab = QWidget()
        layout = QVBoxLayout(self.stt_tab)

        btn_layout = QHBoxLayout()
        self.speech_to_text_btn = QPushButton("🎤 Speech to Text")
        self.speech_to_text_btn.setFixedSize(160, 30)
        self.speech_to_text_btn.clicked.connect(self.handle_speech_to_text)
        btn_layout.addWidget(self.speech_to_text_btn)

        self.mic_select_btn = QPushButton("🎧")
        self.mic_select_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.mic_select_btn.setFixedSize(30, 30)
        self.mic_select_btn.clicked.connect(self.show_microphone_selection)
        btn_layout.addWidget(self.mic_select_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.stt_scroll = QScrollArea()
        self.stt_content = QTextEdit()
        self.stt_content.setReadOnly(True)
        self.stt_content.setText("Speech recognition output will appear here...")
        self.stt_scroll.setWidgetResizable(True)
        self.stt_scroll.setWidget(self.stt_content)
        layout.addWidget(self.stt_scroll)

        self.tabs.addTab(self.stt_tab, "Speech To Text")
    
    # ------------------- Handlers & Model -------------------
    def setup_handlers(self):
        """Initialize functionality handlers in the correct order."""
        # Initialize TTS and STT first
        self.tts_handler = TTSHandler(self)
        self.stt_handler = STTHandler(self, self.username)

        # Initialize model handler BEFORE camera
        self.model_handler = ModelHandler()

        # ✅ Connect prediction signal immediately
        self.model_handler.prediction_made.connect(self.on_model_prediction)
        print("🔌 ModelHandler connected to GUI signal.")

        # Initialize camera handler with model handler
        self.camera_handler = CameraHandler(self.camera_label, self.model_handler)

        # ✅ Start camera AFTER everything is connected
        started = self.camera_handler.start_camera()
        if started:
            print("📸 Camera successfully started with model recognition enabled.")
        else:
            print("❌ Failed to start camera.")
    
    def on_model_prediction(self, label, confidence):
        print(f"🔮 PREDICTION RECEIVED: {label} ({confidence:.2f})")
        self.tts_content.append(f"[Model]: {label} ({confidence:.2f})")

    
    # ------------------- TTS & STT Delegation -------------------
    def handle_text_to_speech(self):
        self.tts_handler.handle_text_to_speech(
            self.tts_content, self.text_to_speech_btn, self.download_audio_btn
        )
    
    def handle_download_audio(self):
        self.tts_handler.handle_download_audio()
    
    def handle_speech_to_text(self):
        self.stt_handler.handle_speech_to_text(
            self.stt_content, self.speech_to_text_btn
        )
    
    def show_microphone_selection(self):
        self.stt_handler.show_microphone_selection()
    
    def clear_placeholder_tts(self, event):
        if self.tts_content.toPlainText() == "Enter text here for Text-To-Speech...":
            self.tts_content.clear()
        QTextEdit.focusInEvent(self.tts_content, event)
    
    def open_preferences(self):
        dialog = PreferencesDialog(self, self.username)
        dialog.exec()
    
    # ------------------- Styling & Preferences -------------------
    def set_dark_mode(self, dark_mode):
        self.dark_mode = dark_mode
        self.apply_styling()
    
    def apply_styling(self):
        colors = apply_theme(self, self.theme_manager, self.dark_mode)
        self.apply_button_styling(colors)
        self.apply_component_styling(colors)
    
    def apply_button_styling(self, colors):
        btn_style = self.theme_manager.get_button_style(colors)
        for btn in [
            self.user_button,
            self.text_to_speech_btn,
            self.download_audio_btn,
            self.speech_to_text_btn,
            self.mic_select_btn
        ]:
            btn.setStyleSheet(btn_style)
    
    def apply_component_styling(self, colors):
        self.top_bar.setStyleSheet(self.theme_manager.get_top_bar_style(colors))
        self.camera_frame.setStyleSheet(self.theme_manager.get_camera_frame_style(colors))
        self.camera_label.setStyleSheet(self.theme_manager.get_camera_label_style(colors))
        self.tabs.setStyleSheet(self.theme_manager.get_tab_widget_style(colors))
        text_edit_style = self.theme_manager.get_text_edit_style(colors)
        self.tts_content.setStyleSheet(text_edit_style)
        self.stt_content.setStyleSheet(text_edit_style)
    
    def load_user_preferences(self):
        prefs = get_user_preferences(self.username)
        self.set_dark_mode(prefs.get("dark_mode", True))
        if self.tts_handler:
            self.tts_handler.update_settings(
                translation=prefs.get("tts_translation", "No Translation"),
                voice=prefs.get("tts_voice", "English (US)"),
                speed=prefs.get("tts_speed", "Normal")
            )
    
    # ------------------- Close Event -------------------
    def closeEvent(self, event):
        if self.camera_handler:
            self.camera_handler.cleanup()
        if self.tts_handler and self.tts_handler.tts_worker:
            self.tts_handler.tts_worker.terminate()
        if self.stt_handler and self.stt_handler.stt_worker:
            self.stt_handler.stt_worker.stop_recording()
            self.stt_handler.stt_worker.wait()
        print("👋 Application closed cleanly.")
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication([])
    username = show_login_flow()
    if username:
        window = EchoMeApp(username)
        window.show()
        app.exec()
    else:
        print("Login cancelled by user")


if __name__ == "__main__":
    main()
