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

# Import our modular components
from user_data import get_user_preferences, save_user_preferences
from tts_handler import TTSHandler
from stt_handler import STTHandler
from preferences_dialog import PreferencesDialog
from styling import ThemeManager, apply_theme
from camera_handler import CameraHandler
from login import show_login_flow


class EchoMeApp(QWidget):
    """Main application window - handles display and UI coordination"""
    
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setup_window()
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        self.dark_mode = True
        
        # Initialize handlers
        self.tts_handler = None
        self.stt_handler = None
        self.camera_handler = None
        
        # UI components
        self.setup_ui()
        self.setup_handlers()
        self.load_user_preferences()
        self.apply_styling()
        
    def setup_window(self):
        """Setup basic window properties"""
        self.setWindowTitle(f"ECHO ME - {self.username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))
        self.setFixedSize(658, 780)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.center_window()
    
    def center_window(self):
        """Center the window on screen"""
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = 50  # Position near top of screen
        self.move(x, y)
    
    def setup_ui(self):
        """Setup the main UI layout"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)
        
        # Create UI sections
        self.create_top_bar()
        self.create_camera_area()
        self.create_tabbed_interface()
    
    def create_top_bar(self):
        """Create the top navigation bar"""
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(80)
        self.top_layout = QHBoxLayout(self.top_bar)
        self.top_layout.setContentsMargins(10, 0, 10, 0)

        # Logo label
        self.logo_label = QLabel("ECHO ME")
        self.logo_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.top_layout.addWidget(self.logo_label)
        self.top_layout.addStretch()

        # User button
        self.user_button = QPushButton(self.username)
        self.user_button.setFixedSize(80, 30)
        self.user_button.clicked.connect(self.open_preferences)
        self.top_layout.addWidget(self.user_button)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.top_bar.setGraphicsEffect(shadow)
        
        self.main_layout.addWidget(self.top_bar)
    
    def create_camera_area(self):
        """Create the camera display area"""
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        self.camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_layout.addWidget(self.camera_label)
        
        self.main_layout.addWidget(self.camera_frame)
    
    def create_tabbed_interface(self):
        """Create the tabbed interface for TTS and STT"""
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Create TTS tab
        self.create_tts_tab()
        
        # Create STT tab
        self.create_stt_tab()
    
    def create_tts_tab(self):
        """Create the Text-to-Speech tab"""
        self.tts_tab = QWidget()
        self.tts_layout = QVBoxLayout(self.tts_tab)

        # TTS buttons layout
        self.tts_button_layout = QHBoxLayout()
        
        self.text_to_speech_btn = QPushButton("🔊 Text to Speech")
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.text_to_speech_btn.clicked.connect(self.handle_text_to_speech)
        self.tts_button_layout.addWidget(self.text_to_speech_btn)

        self.download_audio_btn = QPushButton("📥 Download Audio")
        self.download_audio_btn.setFixedSize(160, 30)
        self.download_audio_btn.setEnabled(False)
        self.download_audio_btn.clicked.connect(self.handle_download_audio)
        self.tts_button_layout.addWidget(self.download_audio_btn)
        
        self.tts_button_layout.addStretch()
        self.tts_layout.addLayout(self.tts_button_layout)

        # TTS text area
        self.tts_scroll = QScrollArea()
        self.tts_content = QTextEdit()
        self.tts_content.setText("Enter text here for Text-To-Speech...")
        self.tts_content.focusInEvent = self.clear_placeholder_tts
        self.tts_scroll.setWidgetResizable(True)
        self.tts_scroll.setWidget(self.tts_content)
        self.tts_layout.addWidget(self.tts_scroll)

        self.tabs.addTab(self.tts_tab, "Text To Speech")
    
    def create_stt_tab(self):
        """Create the Speech-to-Text tab"""
        self.stt_tab = QWidget()
        self.stt_layout = QVBoxLayout(self.stt_tab)

        # STT buttons layout
        self.stt_button_layout = QHBoxLayout()
        
        self.speech_to_text_btn = QPushButton("🎤 Speech to Text")
        self.speech_to_text_btn.setFixedSize(160, 30)
        self.speech_to_text_btn.clicked.connect(self.handle_speech_to_text)
        self.stt_button_layout.addWidget(self.speech_to_text_btn)
        
        # Add microphone selection button
        self.mic_select_btn = QPushButton("🎧")
        self.mic_select_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.mic_select_btn.setFixedSize(30, 30)
        self.mic_select_btn.clicked.connect(self.show_microphone_selection)
        self.mic_select_btn.setToolTip("Select Microphone")
        self.stt_button_layout.addWidget(self.mic_select_btn)
        
        self.stt_button_layout.addStretch()
        self.stt_layout.addLayout(self.stt_button_layout)

        # STT output area
        self.stt_scroll = QScrollArea()
        self.stt_content = QTextEdit()
        self.stt_content.setReadOnly(True)
        self.stt_content.setText("Speech recognition output will appear here...")
        self.stt_scroll.setWidgetResizable(True)
        self.stt_scroll.setWidget(self.stt_content)
        self.stt_layout.addWidget(self.stt_scroll)

        self.tabs.addTab(self.stt_tab, "Speech To Text")
    
    def setup_handlers(self):
        """Initialize functionality handlers"""
        # Initialize TTS handler
        self.tts_handler = TTSHandler(self)
        
        # Initialize STT handler
        self.stt_handler = STTHandler(self, self.username)
        
        # Initialize camera handler
        self.camera_handler = CameraHandler(self.camera_label)
    
    def apply_styling(self):
        """Apply current theme styling to all components"""
        # Get color palette
        colors = apply_theme(self, self.theme_manager, self.dark_mode)
        
        # Apply specific component styling
        self.apply_button_styling(colors)
        self.apply_component_styling(colors)
    
    def apply_button_styling(self, colors):
        """Apply styling to all buttons"""
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
        """Apply styling to UI components"""
        # Top bar styling
        self.top_bar.setStyleSheet(self.theme_manager.get_top_bar_style(colors))
        
        # Camera frame styling
        self.camera_frame.setStyleSheet(self.theme_manager.get_camera_frame_style(colors))
        self.camera_label.setStyleSheet(self.theme_manager.get_camera_label_style(colors))
        
        # Tab widget styling
        self.tabs.setStyleSheet(self.theme_manager.get_tab_widget_style(colors))
        
        # Scroll areas styling
        scroll_style = self.theme_manager.get_scroll_area_style(colors)
        self.tts_scroll.setStyleSheet(scroll_style)
        self.stt_scroll.setStyleSheet(scroll_style)
        
        # Text editors styling
        text_edit_style = self.theme_manager.get_text_edit_style(colors)
        self.tts_content.setStyleSheet(text_edit_style)
        self.stt_content.setStyleSheet(text_edit_style)
    
    # Event handlers that delegate to appropriate modules
    def handle_text_to_speech(self):
        """Handle TTS button click"""
        self.tts_handler.handle_text_to_speech(
            self.tts_content, 
            self.text_to_speech_btn, 
            self.download_audio_btn
        )
    
    def handle_download_audio(self):
        """Handle download audio button click"""
        self.tts_handler.handle_download_audio()
    
    def handle_speech_to_text(self):
        """Handle STT button click"""
        self.stt_handler.handle_speech_to_text(
            self.stt_content,
            self.speech_to_text_btn
        )
    
    def show_microphone_selection(self):
        """Show microphone selection dialog"""
        self.stt_handler.show_microphone_selection()
    
    def open_preferences(self):
        """Open preferences dialog"""
        dialog = PreferencesDialog(self, self.username)
        dialog.exec()
    
    def clear_placeholder_tts(self, event):
        """Clear TTS placeholder text on focus"""
        if self.tts_content.toPlainText() == "Enter text here for Text-To-Speech...":
            self.tts_content.clear()
        QTextEdit.focusInEvent(self.tts_content, event)
    
    def set_dark_mode(self, dark_mode):
        """Set dark/light mode"""
        self.dark_mode = dark_mode
        self.apply_styling()
    
    def landmark_toggle_btn_state(self, state):
        """Toggle landmark visibility"""
        if self.camera_handler:
            self.camera_handler.set_landmark_visibility(state)
    
    def load_user_preferences(self):
        """Load user preferences"""
        prefs = get_user_preferences(self.username)
        
        # Apply preferences
        self.set_dark_mode(prefs.get("dark_mode", True))
        self.landmark_toggle_btn_state(prefs.get("show_landmarks", True))
        
        # Update handlers with preferences
        if self.tts_handler:
            self.tts_handler.update_settings(
                translation=prefs.get("tts_translation", "No Translation"),
                voice=prefs.get("tts_voice", "English (US)"),
                speed=prefs.get("tts_speed", "Normal")
            )
        
        if self.stt_handler:
            self.stt_handler.update_mic_device(prefs.get("mic_device_index", None))
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up resources
        if self.camera_handler:
            self.camera_handler.cleanup()
        
        # Stop any ongoing TTS/STT operations
        if self.tts_handler and self.tts_handler.tts_worker:
            self.tts_handler.tts_worker.terminate()
        
        if self.stt_handler and self.stt_handler.stt_worker:
            self.stt_handler.stt_worker.stop_recording()
            self.stt_handler.stt_worker.wait()
        
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication([])
    
    # Show login flow
    username = show_login_flow()
    
    if username:
        # Create and show main window
        window = EchoMeApp(username)
        window.show()
        
        # Run application
        app.exec()
    else:
        print("Login cancelled by user")


if __name__ == "__main__":
    main()