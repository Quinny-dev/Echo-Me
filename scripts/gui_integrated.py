# ================================================================
# STYLED VERSION - SavedModel Compatible (Keras 2.x)
# Run python scripts/gui_integrated.py    in terminal
# ================================================================
from pathlib import Path
from PySide6.QtWidgets import ( 
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QTextEdit, QFrame, QGraphicsDropShadowEffect,
    QDialog, QFormLayout, QCheckBox, QComboBox, QMessageBox, QFileDialog
)
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtCore import Qt, QThread, Signal
from camera_feed import CameraFeed
from hand_landmarking.hand_landmarking import HandLandmarkDetector
from tts import LANGUAGE_OPTIONS, GTTS_VOICES, convert_and_play, download_audio_files, cleanup
from login import show_login_flow
import tensorflow as tf
import numpy as np
import json
from tools.holistic import normalize_features, to_landmark_row

READY_FILE = Path("gui_ready.flag")
USER_PREF_FILE = Path("user_preferences.json")


def load_user_data():
    if not USER_PREF_FILE.exists():
        return {}
    with open(USER_PREF_FILE, "r") as f:
        return json.load(f)


def save_user_data(data):
    with open(USER_PREF_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_labels_from_data_folder():
    """
    Automatically loads class labels from your data folder.
    Order matters! We use alphabetical folder order just like the training pipeline.
    """
    data_dir = Path("data")
    label_dirs = []
    folder_order = ["None", "holds_data", "nonholds_data"]

    for folder_name in folder_order:
        folder_path = data_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            if folder_name == "None":
                label_dirs.append("None")
            else:
                for f in sorted(folder_path.iterdir()):
                    if f.is_dir():
                        label_dirs.append(f.name)

    print("✅ Loaded labels:", label_dirs)
    return label_dirs


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


# ---- User Preferences Dialog (with full styling) ----
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
        if dark_mode:
            bg_color = "#0d1117"
            card_bg = "#161b22"
            text_color = "#e6edf3"
            accent = "#1f6feb"
            accent_hover = "#58a6ff"
            accent_glow = "rgba(88, 166, 255, 0.4)"
            checkbox_bg = "#161b22"
            combo_bg = "#0d1117"
            border_color = "#30363d"
        else:
            bg_color = "#f6f8fa"
            card_bg = "#ffffff"
            text_color = "#24292f"
            accent = "#0969da"
            accent_hover = "#0550ae"
            accent_glow = "rgba(9, 105, 218, 0.3)"
            checkbox_bg = "#ffffff"
            combo_bg = "#f6f8fa"
            border_color = "#d0d7de"

        self.setStyleSheet(f"""
            QDialog {{ 
                background-color: {bg_color}; 
                color: {text_color}; 
                border-radius: 20px; 
            }}
            QLabel {{ 
                color: {text_color}; 
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-weight: 500;
                letter-spacing: 0.3px;
            }}
            QCheckBox {{ 
                spacing: 12px;
                color: {text_color};
                font-size: 13px;
                font-weight: 500;
                padding: 8px;
                border-radius: 6px;
            }}
            QCheckBox:hover {{
                background-color: {combo_bg};
            }}
            QCheckBox::indicator {{
                width: 20px; 
                height: 20px; 
                border-radius: 6px;
                border: 2px solid {border_color}; 
                background-color: {checkbox_bg};
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 3px {accent_glow};
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent}; 
                border: 2px solid {accent};
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjMzMzMgNEw2IDExLjMzMzNMMi42NjY2NyA4IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
            }}
            QCheckBox::indicator:checked:hover {{
                background-color: {accent_hover};
                border: 2px solid {accent_hover};
            }}
            QPushButton {{
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
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {accent_hover}, stop:1 {accent});
                box-shadow: 0 4px 12px {accent_glow};
            }}
            QPushButton:pressed {{
                background: {accent};
                padding-top: 13px;
                padding-bottom: 11px;
            }}
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
            QComboBox:hover {{
                border: 2px solid {accent};
                background-color: {card_bg};
                box-shadow: 0 0 0 3px {accent_glow};
            }}
            QComboBox:focus {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 3px {accent_glow};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                width: 12px;
                height: 12px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {card_bg};
                color: {text_color};
                selection-background-color: {accent};
                selection-color: white;
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 6px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-radius: 6px;
                margin: 2px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {accent_glow};
            }}
        """)

    def save_preferences(self):
        if self.parent_window and self.user:
            self.parent_window.set_dark_mode(self.dark_mode_checkbox.isChecked())
            self.parent_window.show_landmarks = self.show_landmarks_checkbox.isChecked()
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


# ================================================================
# MAIN APPLICATION WITH STYLING
# ================================================================
class EchoMeApp(QWidget):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle(f"ECHO ME - {username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))
        self.setFixedSize(658, 780)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)

        self.dark_mode = True
        self.show_landmarks = True
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        self.tts_worker = None

        # ------------------------------
        # MAIN LAYOUT
        # ------------------------------
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        # ============================================================
        # TOP BAR
        # ============================================================
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
        self.main_layout.addWidget(self.top_bar)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(20)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.top_bar.setGraphicsEffect(shadow)

        # ============================================================
        # CAMERA AREA
        # ============================================================
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        self.camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_layout.addWidget(self.camera_label)
        self.main_layout.addWidget(self.camera_frame)

        # ============================================================
        # MENU PANEL
        # ============================================================
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

        # ============================================================
        # TRANSCRIPTION PANEL
        # ============================================================
        self.transcription_label = QLabel("Transcription")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setFixedHeight(40)
        self.main_layout.addWidget(self.transcription_label)

        self.scroll_area = QScrollArea()
        self.transcription_content = QTextEdit()
        self.transcription_content.setReadOnly(False)
        self.transcription_content.setText("Transcription goes here...")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.transcription_content)
        self.main_layout.addWidget(self.scroll_area)

        # ============================================================
        # LOAD MODEL - Works with Keras 2.x (your environment)
        # ============================================================
        try:
            self.model_path = "models/model_fast"
            print(f"Loading model from {self.model_path}...")
            
            # Try Keras 2.x method first (what works in your environment)
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                self.TIMESTEPS = self.model.input_shape[1]
                print(f"✅ Model loaded using keras.models.load_model")
            except Exception as e:
                print(f"Keras load_model failed: {e}")
                print("Trying SavedModel approach...")
                
                # Fallback to SavedModel (Keras 3.x)
                loaded_model = tf.saved_model.load(self.model_path)
                self.model = loaded_model.signatures['serving_default']
                
                # Extract input shape
                input_spec = list(self.model.structured_input_signature[1].values())[0]
                self.input_key = list(self.model.structured_input_signature[1].keys())[0]
                self.TIMESTEPS = int(input_spec.shape[1])
                self.is_savedmodel = True
                print(f"✅ Model loaded using tf.saved_model.load")
            else:
                self.is_savedmodel = False
            
            self.LABELS = load_labels_from_data_folder()
            self.window = np.zeros((self.TIMESTEPS, 130), dtype=np.float32)
            self.frame_counter = 0
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

        # ============================================================
        # HAND TRACKING SETUP
        # ============================================================
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        # ============================================================
        # APPLY STYLING
        # ============================================================
        self.load_user_preferences()
        self.apply_styling(self.dark_mode)
        self.signal_ready()

    def process_frame(self, frame, landmarks):
        """Process frame with model - handles both Keras and SavedModel formats"""
        if self.model is None or landmarks is None:
            return
        
        try:
            features = to_landmark_row(landmarks, use_holistic=False)
            normalized = normalize_features(features)

            self.window[:-1] = self.window[1:]
            self.window[-1] = normalized

            self.frame_counter += 1
            if self.frame_counter >= 5:
                self.frame_counter = 0
                
                # Prepare input
                input_data = np.array([self.window], dtype=np.float32)
                
                # Call model based on type
                if hasattr(self, 'is_savedmodel') and self.is_savedmodel:
                    # SavedModel approach (Keras 3.x)
                    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                    preds = self.model(**{self.input_key: input_tensor})
                    output_tensor = list(preds.values())[0]
                    preds_array = output_tensor.numpy()[0]
                else:
                    # Regular Keras model (Keras 2.x)
                    preds = self.model(input_data)
                    if hasattr(preds, 'numpy'):
                        preds_array = preds.numpy()[0]
                    else:
                        preds_array = preds[0]
                
                class_index = int(np.argmax(preds_array))
                confidence = float(np.max(preds_array))
                label = self.LABELS[class_index]

                if confidence > 0.5:
                    self.transcription_content.append(f"[Model]: {label} ({confidence:.2f})")
                    
        except Exception as e:
            print(f"Error during process_frame: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # STYLING METHODS
    # ============================================================
    def apply_styling(self, dark: bool):
        """Applies full application styling based on user preference."""
        self.dark_mode = dark
        
        if dark:
            bg_color = "#0d1117"
            top_bar_color = "#161b22"
            top_bar_gradient = "#1f6feb"
            btn_primary = "#1f6feb"
            btn_hover = "#58a6ff"
            text_color = "#e6edf3"
            border_color = "#30363d"
            frame_bg = "#161b22"
            text_edit_bg = "#0d1117"
        else:
            bg_color = "#f6f8fa"
            top_bar_color = "#ffffff"
            top_bar_gradient = "#0969da"
            btn_primary = "#0969da"
            btn_hover = "#0550ae"
            text_color = "#24292f"
            border_color = "#d0d7de"
            frame_bg = "#ffffff"
            text_edit_bg = "#f6f8fa"

        # Main Window
        self.setStyleSheet(f"QWidget {{ background-color: {bg_color}; color: {text_color}; }}")

        # Top Bar
        self.top_bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1,y2:0,
                    stop:0 {top_bar_color}, stop:0.7 {top_bar_color}, stop:1 {top_bar_gradient});
                border-radius: 18px;
                border: 1px solid {border_color};
            }}
        """)
        self.logo_label.setStyleSheet(f"color: {text_color}; font-weight: bold;")

        # Buttons
        btn_style = f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0,y2:1,
                    stop:0 {btn_primary}, stop:1 {btn_hover});
                color: white;
                border-radius: 10px;
                padding: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0,y2:1,
                    stop:0 {btn_hover}, stop:1 {btn_primary});
            }}
        """
        for btn in [self.user_button, self.text_to_speech_btn, self.download_audio_btn, self.speech_to_text_btn]:
            btn.setStyleSheet(btn_style)

        # Frames
        self.camera_frame.setStyleSheet(f"background-color:{frame_bg}; border-radius:12px; border:1px solid {border_color};")
        self.menu_frame.setStyleSheet(f"background-color:{frame_bg}; border-radius:12px; border:1px solid {border_color};")

        # Transcription
        self.transcription_label.setStyleSheet(f"""
            QLabel {{
                background-color: {frame_bg};
                color: {text_color};
                padding: 10px;
                border-radius: 12px;
                border: 1px solid {border_color};
            }}
        """)
        self.transcription_content.setStyleSheet(f"""
            QTextEdit {{
                background-color: {text_edit_bg};
                color: {text_color};
                border-radius: 12px;
                border: none;
                padding: 12px;
            }}
        """)

    def set_dark_mode(self, enabled: bool):
        """Public method to toggle dark mode"""
        self.apply_styling(enabled)

    # ============================================================
    # TTS HANDLERS
    # ============================================================
    def handle_text_to_speech(self):
        text = self.transcription_content.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Text", "Please enter text to speak.")
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
        if translated_text != self.transcription_content.toPlainText():
            self.transcription_content.append(f"[TTS]: {translated_text}")
        QMessageBox.information(self, "Success", "TTS completed!")

    def on_tts_error(self, error_message):
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("Text to Speech")
        QMessageBox.critical(self, "TTS Error", error_message)

    def handle_download_audio(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Download Folder")
        if folder:
            num = download_audio_files(folder)
            QMessageBox.information(self, "Download Complete", f"Saved {num} files to {folder}")

    # ============================================================
    # UTILITY METHODS
    # ============================================================
    def open_preferences(self):
        dialog = PreferencesDialog(parent=self, user=self.username)
        dialog.exec()

    def signal_ready(self):
        try:
            READY_FILE.write_text("ready")
        except Exception:
            pass

    def load_user_preferences(self):
        data = load_user_data()
        prefs = data.get(self.username, {}).get("preferences", {})
        self.dark_mode = prefs.get("dark_mode", True)
        self.show_landmarks = prefs.get("show_landmarks", True)
        self.tts_translation = prefs.get("tts_translation", "No Translation")
        self.tts_voice = prefs.get("tts_voice", "English (US)")
        self.tts_speed = prefs.get("tts_speed", "Normal")

    def closeEvent(self, event):
        if self.camera:
            self.camera.release()
        if self.hand_detector:
            self.hand_detector.close()
        if READY_FILE.exists():
            READY_FILE.unlink()
        cleanup()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    username = show_login_flow()
    if username:
        window = EchoMeApp(username)
        window.show()
        app.exec()
    else:
        print("Login cancelled by user")