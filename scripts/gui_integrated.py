#-- unstyled version used for testing model 
# -- to run type " python scripts/gui_integrated.py " in console
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
from pathlib import Path
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


class EchoMeApp(QWidget):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle(f"ECHO ME - {username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))
        self.setFixedSize(658, 780)

        self.dark_mode = True
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        self.tts_worker = None

        self.main_layout = QVBoxLayout(self)
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

        # Camera area
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        self.camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_layout.addWidget(self.camera_label)
        self.main_layout.addWidget(self.camera_frame)

        # Transcription panel
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

        # ---- Load Model ----
        self.model_path = "models/model_fast"
        self.model = tf.keras.models.load_model(self.model_path)
        self.LABELS = load_labels_from_data_folder()  # ✅ real labels loaded
        self.TIMESTEPS = self.model.input_shape[1] 
        self.window = np.zeros((self.TIMESTEPS, 130))
        self.frame_counter = 0

        # Hand tracking setup
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        self.show_landmarks = True
        self.signal_ready()
        self.load_user_preferences()
def load_labels_from_data_folder():
    """
    Automatically loads class labels from your data folder.
    Order matters! We use alphabetical folder order just like the training pipeline.
    """
    data_dir = Path("data")
    label_dirs = []

    # These folders contain the label names (based on your directory structure)
    folder_order = ["None", "holds_data", "nonholds_data"]

    for folder_name in folder_order:
        folder_path = data_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            # Add folder itself if it's directly a class (like 'None')
            if folder_name == "None":
                label_dirs.append("None")
            else:
                # Add inner directories as labels
                for f in sorted(folder_path.iterdir()):
                    if f.is_dir():
                        label_dirs.append(f.name)

    print("✅ Loaded labels:", label_dirs)
    return label_dirs


class EchoMeApp(QWidget):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle(f"ECHO ME - {username}")
        self.setWindowIcon(QIcon("assets/Echo_Me_Logo.ico"))
        self.setFixedSize(658, 780)

        self.dark_mode = True
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        self.tts_worker = None

        self.main_layout = QVBoxLayout(self)
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

        # Camera area
        self.camera_frame = QFrame()
        self.camera_frame.setFixedHeight(450)
        self.camera_layout = QVBoxLayout(self.camera_frame)
        self.camera_layout.setAlignment(Qt.AlignCenter)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_layout.addWidget(self.camera_label)
        self.main_layout.addWidget(self.camera_frame)

        # Transcription panel
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

        # ---- Load Model ----
        self.model_path = "models/model_fast"
        self.model = tf.keras.models.load_model(self.model_path)
        self.LABELS = load_labels_from_data_folder()  # ✅ real labels loaded
        self.TIMESTEPS = self.model.input_shape[1]
        self.window = np.zeros((self.TIMESTEPS, 130))
        self.frame_counter = 0

        # Hand tracking setup
        self.hand_detector = HandLandmarkDetector(static_image_mode=False)
        self.camera = CameraFeed(
            self.camera_label,
            frame_callback=self.process_frame,
            hand_detector=self.hand_detector
        )

        self.show_landmarks = True
        self.signal_ready()
        self.load_user_preferences()

    def process_frame(self, frame, landmarks):
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
            print(f"Error during process_frame: {e}")

    def open_preferences(self):
        QMessageBox.information(self, "Preferences", "Preferences dialog coming soon.")

    def signal_ready(self):
        try:
            READY_FILE.write_text("ready")
        except Exception:
            pass

    def load_user_preferences(self):
        data = load_user_data()
        prefs = data.get(self.username, {}).get("preferences", {})
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