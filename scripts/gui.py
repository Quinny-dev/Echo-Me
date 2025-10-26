from PySide6.QtWidgets import ( 
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QDialog, QFormLayout, QCheckBox, QComboBox, QLineEdit, QMessageBox,
    QFileDialog, QTabWidget, QStyle
)
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtCore import Qt, QThread, Signal
from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from tts import LANGUAGE_OPTIONS, GTTS_VOICES, convert_and_play, download_audio_files, cleanup
from login import show_login_flow
from pathlib import Path
import json
import speech_recognition as sr
import queue

READY_FILE = Path("gui_ready.flag")
USER_PREF_FILE = Path("user_preferences.json")

def load_user_data():
    with open(USER_PREF_FILE, "r") as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_PREF_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ---- TTS Worker Thread ----
class TTSWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, text, preferences):
        super().__init__()
        self.text = text
        self.preferences = preferences
    
    def run(self):
        try:
            translated_text = convert_and_play(self.text, self.preferences)
            self.finished.emit(translated_text)
        except Exception as e:
            self.error.emit(str(e))

# ---- STT Worker Thread ----
class STTWorker(QThread):
    text_recognized = Signal(str)
    error = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, mic_device_index=None):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.mic_device_index = mic_device_index
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def run(self):
        try:
            if self.mic_device_index is None:
                mic = sr.Microphone()
            else:
                mic = sr.Microphone(device_index=self.mic_device_index)
            
            self.status_update.emit("🎤 Adjusting for ambient noise...")
            
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.status_update.emit("🎤 Listening... Speak now!")
            self.is_recording = True
            
            stop_listening = self.recognizer.listen_in_background(mic, self._audio_callback)
            
            while self.is_recording:
                self.msleep(100)
                
                try:
                    while True:
                        text = self.audio_queue.get_nowait()
                        self.text_recognized.emit(text)
                except queue.Empty:
                    pass
            
            stop_listening(wait_for_stop=False)
            
        except Exception as e:
            self.error.emit(f"Microphone error: {str(e)}")
    
    def _audio_callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio)
            self.audio_queue.put(text)
        except sr.UnknownValueError:
            self.audio_queue.put("⚠️ Could not understand audio")
        except sr.RequestError as e:
            self.audio_queue.put(f"API error: {e}")
    
    def stop_recording(self):
        self.is_recording = False

# ---- User Preferences Dialog ----
class PreferencesDialog(QDialog):
    def __init__(self, parent=None, user=None):
        super().__init__(parent)
        self.setWindowTitle("User Preferences")
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(450, 400)

        self.parent_window = parent
        self.user = user
        user_data = load_user_data().get(user, {}).get("preferences", {})

        self.dark_mode_checkbox = QCheckBox("Enable Dark Mode")
        self.dark_mode_checkbox.setChecked(user_data.get("dark_mode", True))

        self.show_landmarks_checkbox = QCheckBox("Show landmark overlay")
        self.show_landmarks_checkbox.setChecked(user_data.get("show_landmarks", True))

        self.translation_combo = QComboBox()
        self.translation_combo.addItems(list(LANGUAGE_OPTIONS.keys()))
        self.translation_combo.setCurrentText(user_data.get("tts_translation", "No Translation"))

        self.voice_combo = QComboBox()
        self.voice_combo.addItems(list(GTTS_VOICES.keys()))
        self.voice_combo.setCurrentText(user_data.get("tts_voice", "English (US)"))

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Slow", "Normal", "Fast"])
        self.speed_combo.setCurrentText(user_data.get("tts_speed", "Normal"))

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_preferences)

        layout = QFormLayout(self)
        layout.addRow(self.dark_mode_checkbox)
        layout.addRow(self.show_landmarks_checkbox)
        layout.addRow(QLabel("Translate to:"), self.translation_combo)
        layout.addRow(QLabel("Voice:"), self.voice_combo)
        layout.addRow(QLabel("Speed:"), self.speed_combo)
        layout.addRow(save_btn)

        self.apply_styles(self.dark_mode_checkbox.isChecked())

    def apply_styles(self, dark_mode):
        # ---- Color Palette Configuration ----
        # Define all colors based on dark/light mode preference
        if dark_mode:
            # Dark mode: GitHub-inspired dark theme with blue accents
            bg_color = "#0d1117"              # Main dialog background
            card_bg = "#161b22"               # Card/panel backgrounds
            text_color = "#e6edf3"            # Primary text color
            accent = "#1f6feb"                # Primary accent/brand color
            accent_hover = "#58a6ff"          # Lighter accent for hover states
            accent_glow = "rgba(88, 166, 255, 0.4)"  # Glow effect color
            checkbox_bg = "#161b22"           # Checkbox background
            combo_bg = "#0d1117"              # Dropdown background
            border_color = "#30363d"          # Subtle border color
            shadow_color = "rgba(0, 0, 0, 0.6)"  # Shadow for depth
        else:
            # Light mode: Clean, professional palette with blue accents
            bg_color = "#f6f8fa"              # Main dialog background
            card_bg = "#ffffff"               # Card/panel backgrounds
            text_color = "#24292f"            # Primary text color
            accent = "#0969da"                # Primary accent/brand color
            accent_hover = "#0550ae"          # Darker accent for hover states
            accent_glow = "rgba(9, 105, 218, 0.3)"  # Glow effect color
            checkbox_bg = "#ffffff"           # Checkbox background
            combo_bg = "#f6f8fa"              # Dropdown background
            border_color = "#d0d7de"          # Subtle border color
            shadow_color = "rgba(0, 0, 0, 0.08)"  # Light shadow for depth

        self.setStyleSheet(f"""
            /* ---- Dialog Container ---- */
            /* Main preferences dialog window styling */
            QDialog {{ 
                background-color: {bg_color}; 
                color: {text_color}; 
                border-radius: 20px; 
            }}
            
            /* ---- Labels ---- */
            /* Form labels and text displays */
            QLabel {{ 
                color: {text_color}; 
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-weight: 500;
                letter-spacing: 0.3px;  /* Improved readability */
            }}
            
            /* ---- Frame Containers ---- */
            /* Generic frame/panel styling (currently unused in this dialog) */
            QFrame {{
                background-color: {card_bg};
                border-radius: 12px;
                padding: 15px;
                border: 1px solid {border_color};
            }}
            
            /* ---- Checkboxes ---- */
            /* Checkbox text and container */
            QCheckBox {{ 
                spacing: 12px;              /* Space between checkbox and label */
                color: {text_color};
                font-size: 13px;
                font-weight: 500;
                padding: 8px;
                border-radius: 6px;
            }}
            /* Hover effect for entire checkbox area */
            QCheckBox:hover {{
                background-color: {combo_bg};
            }}
            
            /* Checkbox indicator box (the actual checkbox square) */
            QCheckBox::indicator {{
                width: 20px; 
                height: 20px; 
                border-radius: 6px;
                border: 2px solid {border_color}; 
                background-color: {checkbox_bg};
                transition: all 0.2s ease;
            }}
            /* Checkbox hover state with glow effect */
            QCheckBox::indicator:hover {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 3px {accent_glow};  /* Glow ring effect */
            }}
            /* Checked state with checkmark icon */
            QCheckBox::indicator:checked {{
                background-color: {accent}; 
                border: 2px solid {accent};
                /* SVG checkmark embedded as base64 */
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjMzMzMgNEw2IDExLjMzMzNMMi42NjY2NyA4IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }}
            /* Checked checkbox hover state */
            QCheckBox::indicator:checked:hover {{
                background-color: {accent_hover};
                border: 2px solid {accent_hover};
            }}
            
            /* ---- Buttons ---- */
            /* Primary action buttons (e.g., Save button) */
            QPushButton {{
                /* Horizontal gradient from accent to hover color */
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {accent}, stop:1 {accent_hover});
                color: white; 
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 12px 24px;
                border: none;
                letter-spacing: 0.5px;
            }}
            /* Button hover state with reversed gradient and glow */
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {accent_hover}, stop:1 {accent});
                box-shadow: 0 4px 12px {accent_glow};  /* Elevated glow effect */
            }}
            /* Button pressed/clicked state */
            QPushButton:pressed {{
                background: {accent};
                padding-top: 13px;          /* Subtle downward press effect */
                padding-bottom: 11px;
            }}
            
            /* ---- Dropdown/Combo Boxes ---- */
            /* Main dropdown styling */
            QComboBox {{
                background-color: {combo_bg};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: 500;
                min-height: 28px;
            }}
            /* Dropdown hover state */
            QComboBox:hover {{
                border: 2px solid {accent};
                background-color: {card_bg};
                box-shadow: 0 0 0 3px {accent_glow};  /* Focus ring effect */
            }}
            /* Dropdown focus state (when clicked/opened) */
            QComboBox:focus {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 3px {accent_glow};
            }}
            /* Dropdown arrow button area */
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
                width: 20px;
            }}
            /* Dropdown arrow icon */
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            
            /* ---- Dropdown Menu List ---- */
            /* The popup list that appears when dropdown is opened */
            QComboBox QAbstractItemView {{
                background-color: {card_bg};
                color: {text_color};
                selection-background-color: {accent};  /* Selected item background */
                selection-color: white;                /* Selected item text */
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 6px;
                outline: none;
            }}
            /* Individual items in dropdown list */
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
            }}
            /* Item hover state in dropdown list */
            QComboBox QAbstractItemView::item:hover {{
                background-color: {accent_glow};
            }}
        """)

    def save_preferences(self):
        if self.parent_window and self.user:
            self.parent_window.landmark_toggle_btn_state(self.show_landmarks_checkbox.isChecked())
            self.parent_window.set_dark_mode(self.dark_mode_checkbox.isChecked())
            self.parent_window.tts_translation = self.translation_combo.currentText()
            self.parent_window.tts_voice = self.voice_combo.currentText()
            self.parent_window.tts_speed = self.speed_combo.currentText()

            data = load_user_data()
            data[self.user]["preferences"] = {
                "dark_mode": self.dark_mode_checkbox.isChecked(),
                "show_landmarks": self.show_landmarks_checkbox.isChecked(),
                "tts_translation": self.translation_combo.currentText(),
                "tts_voice": self.voice_combo.currentText(),
                "tts_speed": self.speed_combo.currentText()
            }
            save_user_data(data)
        self.accept()

# ---- Main App ----
class EchoMeApp(QWidget):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle(f"ECHO ME - {username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))

        self.setFixedSize(658, 780)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self.dark_mode = True
        self.center_top()

        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        self.tts_worker = None
        
        # STT variables
        self.stt_worker = None
        self.stt_is_recording = False
        self.mic_device_index = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        # ---- Top bar ----
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(80)
        self.top_layout = QHBoxLayout(self.top_bar)
        self.top_layout.setContentsMargins(10, 0, 10, 0)

        self.logo_label = QLabel("ECHO ME")
        self.logo_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.top_layout.addWidget(self.logo_label)
        self.top_layout.addStretch()

        self.user_button = QPushButton(self.username)
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

        # ---- Tabbed Interface ----
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Text-to-Speech Tab
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

        self.tts_scroll = QScrollArea()
        self.tts_content = QTextEdit()
        self.tts_content.setText("Enter text here for Text-To-Speech...")
        self.tts_content.focusInEvent = self.clear_placeholder_tts
        self.tts_scroll.setWidgetResizable(True)
        self.tts_scroll.setWidget(self.tts_content)
        self.tts_layout.addWidget(self.tts_scroll)

        self.tabs.addTab(self.tts_tab, "Text To Speech")

        # Speech-to-Text Tab
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
        
        # Add stretch to push buttons to the left
        self.stt_button_layout.addStretch()
        
        # Add the horizontal layout to the main vertical layout
        self.stt_layout.addLayout(self.stt_button_layout)

        self.stt_scroll = QScrollArea()
        self.stt_content = QTextEdit()
        self.stt_content.setReadOnly(False)
        self.stt_content.setText("Speech recognition output will appear here...")
        self.stt_content.setReadOnly(True)  # STT content should be read-only
        self.stt_scroll.setWidgetResizable(True)
        self.stt_scroll.setWidget(self.stt_content)
        self.stt_layout.addWidget(self.stt_scroll)

        self.tabs.addTab(self.stt_tab, "Speech To Text")

        # ---- Hand Detector & Camera ----
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        self.show_landmarks = True
        self.signal_ready()
        self.set_dark_mode(self.dark_mode)
        self.load_user_preferences()

    def clear_placeholder_tts(self, event):
        if self.tts_content.toPlainText() == "Enter text here for Text-To-Speech...":
            self.tts_content.clear()
        QTextEdit.focusInEvent(self.tts_content, event)

    def set_dark_mode(self, dark: bool):
        self.dark_mode = dark
        
        # ---- Color Palette Configuration ----
        # Define all colors based on dark/light mode preference
        if dark:
            # Dark mode: GitHub-inspired dark theme with blue accents
            bg_color = "#0d1117"                      # Main window background
            top_bar_color = "#161b22"                 # Top bar base color
            top_bar_gradient = "#1f6feb"              # Top bar gradient accent
            btn_primary = "#1f6feb"                   # Primary button color
            btn_hover = "#58a6ff"                     # Button hover state
            btn_glow = "rgba(88, 166, 255, 0.4)"      # Button glow effect
            transcription_bg = "#161b22"              # Transcription label background
            text_color = "#e6edf3"                    # All text color
            camera_bg = "#161b22"                     # Camera frame background
            menu_bg = "#161b22"                       # Menu panel background
            scroll_bg = "#161b22"                     # Scroll area background
            text_edit_bg = "#0d1117"                  # Text editor background
            border_color = "#30363d"                  # Border color for all elements
            shadow_dark = "rgba(0, 0, 0, 0.6)"        # Strong shadow for elevation
            shadow_light = "rgba(0, 0, 0, 0.3)"       # Light shadow for subtle depth
        else:
            # Light mode: Clean, professional palette with blue accents
            bg_color = "#f6f8fa"                      # Main window background
            top_bar_color = "#ffffff"                 # Top bar base color
            top_bar_gradient = "#0969da"              # Top bar gradient accent
            btn_primary = "#0969da"                   # Primary button color
            btn_hover = "#0550ae"                     # Button hover state
            btn_glow = "rgba(9, 105, 218, 0.3)"       # Button glow effect
            transcription_bg = "#ffffff"              # Transcription label background
            text_color = "#24292f"                    # All text color
            camera_bg = "#ffffff"                     # Camera frame background
            menu_bg = "#ffffff"                       # Menu panel background
            scroll_bg = "#ffffff"                     # Scroll area background
            text_edit_bg = "#f6f8fa"                  # Text editor background
            border_color = "#d0d7de"                  # Border color for all elements
            shadow_dark = "rgba(0, 0, 0, 0.15)"       # Strong shadow for elevation
            shadow_light = "rgba(0, 0, 0, 0.08)"      # Light shadow for subtle depth

        # ---- Main Window Background ----
        # Base styling for entire application window
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            }}
        """)
        
        # ---- Top Bar Header ----
        # Header bar with logo and user button, includes gradient effect
        self.top_bar.setStyleSheet(f"""
            QFrame {{
                /* Horizontal gradient from solid color to accent */
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {top_bar_color}, stop:0.7 {top_bar_color}, stop:1 {top_bar_gradient});
                border-radius: 18px;
                border: 1px solid {border_color};
            }}
        """)
        
        # ---- Logo Text ----
        # "ECHO ME" branding text in top bar
        self.logo_label.setStyleSheet(f"""
            color: {text_color};
            font-weight: 700;              /* Extra bold for prominence */
            letter-spacing: 1px;           /* Spaced out for brand effect */
        """)

        # ---- Button Styling ----
        # Universal button style for all buttons (user, TTS, download, STT)
        btn_style = f"""
            QPushButton {{
                /* Vertical gradient from primary to hover color */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {btn_primary}, stop:1 {btn_hover});
                color: white;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                border: none;
                letter-spacing: 0.3px;     /* Slight spacing for readability */
            }}
            /* Hover state with reversed gradient and glow */
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {btn_hover}, stop:1 {btn_primary});
                box-shadow: 0 4px 12px {btn_glow};  /* Elevated shadow effect */
            }}
            /* Pressed/clicked state */
            QPushButton:pressed {{
                background: {btn_hover};
                padding-top: 1px;          /* Subtle downward press animation */
            }}
            /* Disabled state (e.g., Download Audio when inactive) */
            QPushButton:disabled {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4a4a4a, stop:1 #3a3a3a);
                color: #888888;            /* Muted text for disabled state */
            }}
        """

        # ---- Apply Button Styles ----
        # Apply the unified button styling to all button widgets
        for btn in [
            self.user_button,           # Username button in top bar
            self.text_to_speech_btn,    # Text to Speech button
            self.download_audio_btn,    # Download Audio button
            self.speech_to_text_btn,    # Speech to Text button
            self.mic_select_btn         # Microphone selection button
        ]:
            btn.setStyleSheet(btn_style)

        # ---- Camera Frame Container ----
        # Container that holds the camera feed display
        self.camera_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {camera_bg}; 
                border-radius: 18px;
                border: 1px solid {border_color};
            }}
        """)
        
        # ---- Tab Widget ----
        # Tab container styling
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {scroll_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QTabBar::tab {{
                background-color: {camera_bg};
                color: {text_color};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {border_color};
            }}
            QTabBar::tab:selected {{
                background-color: {btn_primary};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {btn_hover};
                color: white;
            }}
        """)
        
        # ---- Camera Display Label ----
        # The actual video feed display area with black background
        self.camera_label.setStyleSheet(f"""
            QLabel {{
                background-color: #000000;  /* Black for video feed */
                border-radius: 14px;
                border: 2px solid {border_color};
            }}
        """)
        
        # ---- Scroll Area Containers ----
        # Scrollable containers for TTS and STT text editors
        scroll_style = f"""
            QScrollArea {{
                background-color: {scroll_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            
            /* ---- Custom Scrollbar ---- */
            /* Vertical scrollbar track */
            QScrollBar:vertical {{
                background: {text_edit_bg};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            /* Scrollbar handle (draggable part) */
            QScrollBar::handle:vertical {{
                background: {btn_primary};
                border-radius: 5px;
                min-height: 30px;
            }}
            /* Scrollbar handle hover state */
            QScrollBar::handle:vertical:hover {{
                background: {btn_hover};
            }}
            /* Hide scrollbar arrow buttons */
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
        
        # Apply scroll styling to both scroll areas
        self.tts_scroll.setStyleSheet(scroll_style)
        self.stt_scroll.setStyleSheet(scroll_style)
        
        # ---- Text Editor Styling ----
        # Common styling for text editing areas
        text_edit_style = f"""
            QTextEdit {{
                background-color: {text_edit_bg};
                color: {text_color};
                border: none;
                border-radius: 10px;
                padding: 16px;
                font-size: 11pt;
                line-height: 1.6;          /* Comfortable line spacing */
                font-weight: 400;          /* Regular weight for body text */
                selection-background-color: {btn_primary};  /* Highlighted text background */
                selection-color: white;                      /* Highlighted text color */
            }}
        """
        
        # Apply text editor styling to both content areas
        self.tts_content.setStyleSheet(text_edit_style)
        self.stt_content.setStyleSheet(text_edit_style)

    def open_preferences(self):
        dialog = PreferencesDialog(self, self.username)
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
        cleanup()
        event.accept()

    def load_user_preferences(self):
        data = load_user_data()
        prefs = data.get(self.username, {}).get("preferences", {})
        self.set_dark_mode(prefs.get("dark_mode", True))
        self.landmark_toggle_btn_state(prefs.get("show_landmarks", True))
        self.tts_translation = prefs.get("tts_translation", "No Translation")
        self.tts_voice = prefs.get("tts_voice", "English (US)")
        self.tts_speed = prefs.get("tts_speed", "Normal")
        self.mic_device_index = prefs.get("mic_device_index", None)
        
    def handle_text_to_speech(self):
        text = self.tts_content.toPlainText().strip()
        
        if not text or text == "Enter text here for Text-To-Speech...":
            QMessageBox.warning(self, "No Text", "Please enter text in the TTS area first.")
            return

        prefs_for_tts = {
            "translate_to": self.tts_translation,
            "voice": self.tts_voice,
            "speed": self.tts_speed
        }

        self.text_to_speech_btn.setEnabled(False)
        self.text_to_speech_btn.setText("Processing...")
        self.download_audio_btn.setEnabled(False)

        self.tts_worker = TTSWorker(text, prefs_for_tts)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.error.connect(self.on_tts_error)
        self.tts_worker.start()

    def on_tts_finished(self, translated_text):
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("🔊 Text to Speech")
        self.download_audio_btn.setEnabled(True)
        
        original_text = self.tts_content.toPlainText().strip()
        
        if translated_text != original_text and self.tts_translation != "No Translation":
            self.tts_content.setText(translated_text)
        
        QMessageBox.information(self, "Playback Complete", "Audio playback finished successfully!")

    def on_tts_error(self, error_message):
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("🔊 Text to Speech")
        QMessageBox.critical(self, "TTS Error", f"Text-to-Speech failed:\n{error_message}")

    def handle_download_audio(self):
        folder_selected = QFileDialog.getExistingDirectory(
            self, 
            "Select Download Folder",
            str(Path.home() / "Downloads")
        )
        
        if not folder_selected:
            return

        try:
            num_files = download_audio_files(folder_selected)
            QMessageBox.information(
                self, 
                "Download Complete", 
                f"Successfully saved {num_files} audio file(s) to:\n{folder_selected}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Failed to download audio files:\n{str(e)}")

    def handle_speech_to_text(self):
        """Start or stop speech-to-text recording"""
        if not self.stt_is_recording:
            # Start recording
            self.stt_content.clear()
            self.stt_content.append("🎤 Starting speech recognition...\n")
            
            self.speech_to_text_btn.setText("⏹ Stop Recording")
            self.speech_to_text_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    border-radius: 5px;
                    height: 30px;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            
            self.stt_is_recording = True
            
            # Create and start STT worker
            self.stt_worker = STTWorker(self.mic_device_index)
            self.stt_worker.text_recognized.connect(self.on_stt_text_recognized)
            self.stt_worker.status_update.connect(self.on_stt_status_update)
            self.stt_worker.error.connect(self.on_stt_error)
            self.stt_worker.start()
            
        else:
            # Stop recording
            self.stop_stt_recording()

    def stop_stt_recording(self):
        """Stop STT recording and reset button"""
        if self.stt_worker:
            self.stt_worker.stop_recording()
            self.stt_worker.wait()
            
        self.stt_is_recording = False
        self.speech_to_text_btn.setText("🎤 Speech to Text")
        
        # Reset button style based on theme
        if self.dark_mode:
            btn_color = "#00b3b3"
            hover_color = "#66ffff"
        else:
            btn_color = "#008080"
            hover_color = "#00cccc"
            
        self.speech_to_text_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_color};
                color: white;
                border-radius: 5px;
                height: 30px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)

    def on_stt_text_recognized(self, text):
        """Handle recognized text from STT"""
        current_text = self.stt_content.toPlainText()
        if "🎤 Starting speech recognition..." in current_text:
            self.stt_content.clear()
        
        if not text.startswith("⚠️") and not text.startswith("API error"):
            self.stt_content.append(f"{text}")
        else:
            self.stt_content.append(f"<i>{text}</i>")

    def on_stt_status_update(self, status):
        """Handle STT status updates"""
        current_text = self.stt_content.toPlainText()
        if "🎤 Starting speech recognition..." in current_text:
            self.stt_content.clear()
        self.stt_content.append(f"<i>{status}</i>")

    def on_stt_error(self, error_message):
        """Handle STT errors"""
        self.stop_stt_recording()
        QMessageBox.critical(self, "Speech Recognition Error", f"Speech-to-Text failed:\n{error_message}")

    def show_microphone_selection(self):
        """Show microphone selection dialog"""
        try:
            # Get list of available microphones
            mic_names = sr.Microphone.list_microphone_names()
            
            if not mic_names:
                QMessageBox.warning(self, "No Microphones", "No microphones found on this system.")
                return
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setFixedSize(400, 300)
            dialog.setWindowTitle("Select Microphone")
            dialog.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
            
            layout = QVBoxLayout(dialog)
            
            label = QLabel("Select your microphone:")
            layout.addWidget(label)
            
            # Add microphone list
            mic_combo = QComboBox()
            mic_combo.addItems(mic_names)
            
            # Select currently selected microphone
            if self.mic_device_index is not None and self.mic_device_index < len(mic_names):
                mic_combo.setCurrentIndex(self.mic_device_index)
            
            layout.addWidget(mic_combo)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            
            def on_ok():
                self.mic_device_index = mic_combo.currentIndex()
                selected_name = mic_combo.currentText()
                
                # Save to user preferences
                data = load_user_data()
                if self.username not in data:
                    data[self.username] = {"preferences": {}}
                elif "preferences" not in data[self.username]:
                    data[self.username]["preferences"] = {}
                
                data[self.username]["preferences"]["mic_device_index"] = self.mic_device_index
                save_user_data(data)
                
                dialog.accept()
                QMessageBox.information(
                    self, 
                    "Microphone Selected", 
                    f"Selected microphone:\n{selected_name}"
                )
            
            def on_cancel():
                dialog.reject()
            
            ok_btn.clicked.connect(on_ok)
            cancel_btn.clicked.connect(on_cancel)
            
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get microphone list:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication([])

    username = show_login_flow()
    
    if username:
        window = EchoMeApp(username)
        window.show()
        app.exec()
    else:
        print("Login cancelled by user")