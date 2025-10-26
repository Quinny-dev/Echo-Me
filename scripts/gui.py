# ================================================================
# STYLED VERSION - No current model built for  (Keras 3.x)
# to run double click executable on screen
# ================================================================
import tensorflow as tf
import numpy as np
from tools.holistic import normalize_features, to_landmark_row
from pathlib import Path
from PySide6.QtWidgets import ( 
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QDialog, QFormLayout, QCheckBox, QComboBox, QLineEdit, QMessageBox,
    QFileDialog
)
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtCore import Qt, QThread, Signal
from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from tts import LANGUAGE_OPTIONS, GTTS_VOICES, convert_and_play, download_audio_files, cleanup
from login import show_login_flow
import json

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

def load_labels_from_data_folder():
    """
    Automatically loads class labels from your data folder in alphabetical order.
    Matches training pipeline logic.
    """
    data_dir = Path("data")
    label_dirs = []

    folder_order = ["None", "holds_data", "nonholds_data"]

    for folder_name in folder_order:
        folder_path = data_dir / folder_name
        if folder_name == "None":
            label_dirs.append("None")
        else:
            for f in sorted(folder_path.iterdir()):
                if f.is_dir():
                    label_dirs.append(f.name)

    print("✅ Loaded labels:", label_dirs)
    return label_dirs        
class ModelLoader(QThread):
    model_loaded = Signal(object, list, int)  # model, labels, timesteps
    error = Signal(str)

    def run(self):
        try:
            model_path = "models/model_fast"
            model = tf.keras.models.load_model(model_path)
            labels = load_labels_from_data_folder()
            timesteps = model.input_shape[1]
            self.model_loaded.emit(model, labels, timesteps)
        except Exception as e:
            self.error.emit(str(e))

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

        # ---- User TTS Settings ----
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        self.tts_worker = None

        # ---- Main Layout ----
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

        # ---- Loading Placeholder ----
        self.model = None
        self.LABELS = []
        self.TIMESTEPS = 0
        self.window = None
        self.frame_counter = 0

        self.loading_label = QLabel("Loading model... Please wait")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.main_layout.addWidget(self.loading_label)

        self.model_loader = ModelLoader()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.error.connect(self.on_model_error)
        self.model_loader.start()

        # ---- Menu Panel ----
        self.menu_frame = QFrame()
        self.menu_frame.setFixedHeight(60)
        self.menu_layout = QHBoxLayout(self.menu_frame)
        self.menu_layout.setContentsMargins(10, 10, 10, 10)
        self.menu_layout.setSpacing(20)

        self.text_to_speech_btn = QPushButton("Text to Speech")
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.text_to_speech_btn.clicked.connect(self.handle_text_to_speech)
        self.menu_layout.addWidget(self.text_to_speech_btn, alignment=Qt.AlignLeft)

        self.download_audio_btn = QPushButton("Download Audio")
        self.download_audio_btn.setFixedSize(160, 30)
        self.download_audio_btn.setEnabled(False)
        self.download_audio_btn.clicked.connect(self.handle_download_audio)
        self.menu_layout.addWidget(self.download_audio_btn, alignment=Qt.AlignCenter)

        self.speech_to_text_btn = QPushButton("Speech to Text")
        self.speech_to_text_btn.setFixedSize(160, 30)
        self.menu_layout.addWidget(self.speech_to_text_btn, alignment=Qt.AlignRight)

        self.main_layout.addWidget(self.menu_frame)

        # ---- Transcription Panel ----
        self.transcription_label = QLabel("Transcription")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setFixedHeight(40)
        self.main_layout.addWidget(self.transcription_label)

        self.scroll_area = QScrollArea()
        self.transcription_content = QTextEdit()
        self.transcription_content.setReadOnly(False)
        self.transcription_content.setText("Transcription goes here...")
        self.transcription_content.focusInEvent = self.clear_placeholder
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

        self.show_landmarks = True
        self.signal_ready()
        self.set_dark_mode(self.dark_mode)
        self.load_user_preferences()

    # ------------------------ MODEL CALLBACKS ------------------------

    def on_model_loaded(self, model, labels, timesteps):
        self.model = model
        self.LABELS = labels
        self.TIMESTEPS = timesteps
        self.window = np.zeros((self.TIMESTEPS, 130))  # ✅ Your confirmed vector size

        if self.loading_label:
            self.loading_label.deleteLater()

        self.transcription_content.append("[System]: ✅ Model loaded successfully!")
        print("✅ Model is ready and prediction will start automatically.")

    def on_model_error(self, error_message):
        QMessageBox.critical(self, "Model Load Error", f"Failed to load model:\n{error_message}")
        print(f"❌ Model load failed: {error_message}")

    # ------------------------ FRAME PROCESSING ------------------------

    def process_frame(self, frame, landmarks):
        if self.model is None or self.window is None:
            return  # Model not ready yet

        if landmarks is None:
            return

        try:
            features = to_landmark_row(landmarks, use_holistic=False)
            normalized = normalize_features(features)
            self.window[:-1] = self.window[1:]
            self.window[-1] = normalized

            self.frame_counter += 1
            if self.frame_counter >= 5:
                self.frame_counter = 0
                preds = self.model(np.array([self.window]))
                class_index = int(np.argmax(preds))
                confidence = float(np.max(preds))
                label = self.LABELS[class_index]

                if confidence > 0.5:
                    self.transcription_content.append(f"[Model]: {label} ({confidence:.2f})")

        except Exception as e:
            print(f"⚠️ Error during frame processing: {e}")

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
        
    def handle_text_to_speech(self):
        text = self.transcription_content.toPlainText().strip()
        
        if not text or text == "Transcription goes here...":
            QMessageBox.warning(self, "No Text", "Please enter text in the transcription area first.")
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
        self.text_to_speech_btn.setText("Text to Speech")
        self.download_audio_btn.setEnabled(True)
        
        original_text = self.transcription_content.toPlainText().strip()
        
        if translated_text != original_text and self.tts_translation != "No Translation":
            self.transcription_content.setText(translated_text)
        
        QMessageBox.information(self, "Playback Complete", "Audio playback finished successfully!")

    def on_tts_error(self, error_message):
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("Text to Speech")
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


if __name__ == "__main__":
    app = QApplication([])

    username = show_login_flow()
    
    if username:
        window = EchoMeApp(username)
        window.show()
        app.exec()
    else:
        print("Login cancelled by user")