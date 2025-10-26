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
    finished = Signal(str)  # emits translated text
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
    text_recognized = Signal(str)  # emits recognized text
    error = Signal(str)
    status_update = Signal(str)  # for status updates
    
    def __init__(self, mic_device_index=None):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.mic_device_index = mic_device_index
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def run(self):
        try:
            # Select microphone
            if self.mic_device_index is None:
                mic = sr.Microphone()
            else:
                mic = sr.Microphone(device_index=self.mic_device_index)
            
            self.status_update.emit("🎤 Adjusting for ambient noise...")
            
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.status_update.emit("🎤 Listening... Speak now!")
            self.is_recording = True
            
            # Start listening in background
            stop_listening = self.recognizer.listen_in_background(mic, self._audio_callback)
            
            # Keep thread alive while recording
            while self.is_recording:
                self.msleep(100)  # Check every 100ms
                
                # Process audio queue
                try:
                    while True:
                        text = self.audio_queue.get_nowait()
                        self.text_recognized.emit(text)
                except queue.Empty:
                    pass
            
            # Stop listening
            stop_listening(wait_for_stop=False)
            
        except Exception as e:
            self.error.emit(f"Microphone error: {str(e)}")
    
    def _audio_callback(self, recognizer, audio):
        """Callback for background listening"""
        try:
            text = recognizer.recognize_google(audio)
            self.audio_queue.put(text)
        except sr.UnknownValueError:
            self.audio_queue.put("⚠️ Could not understand audio")
        except sr.RequestError as e:
            self.audio_queue.put(f"API error: {e}")
    
    def stop_recording(self):
        """Stop the recording"""
        self.is_recording = False

# ---- User Preferences Dialog ----
class PreferencesDialog(QDialog):
    def __init__(self, parent=None, user=None):
        super().__init__(parent)
        self.setWindowTitle("User Preferences")
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(400, 350)

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
        
        self.stt_worker = None
        self.stt_is_recording = False
        self.mic_device_index = None

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

        # ---- Transcription panel ----

        self.transcription_label = QLabel("Transcription")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setFixedHeight(40)
        self.main_layout.addWidget(self.transcription_label)

        self.tabs = QTabWidget()

        self.tts_tab = QWidget()
        self.tts_layout = QVBoxLayout(self.tts_tab)

        # Create horizontal layout for buttons
        self.tts_button_layout = QHBoxLayout()
        
        self.text_to_speech_btn = QPushButton("Text to Speech")
        self.text_to_speech_btn.setFixedSize(160, 30)
        self.text_to_speech_btn.clicked.connect(self.handle_text_to_speech)
        self.tts_button_layout.addWidget(self.text_to_speech_btn)

        self.download_audio_btn = QPushButton("")
        # Use standard save icon (works on all platforms)
        self.download_audio_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.download_audio_btn.setFixedSize(30, 30)
        self.download_audio_btn.setEnabled(False)  # Disabled until TTS completes
        self.download_audio_btn.clicked.connect(self.handle_download_audio)
        self.download_audio_btn.setToolTip("Download Audio")  # Add tooltip for clarity
        self.tts_button_layout.addWidget(self.download_audio_btn)
        
        # Add stretch to push buttons to the left
        self.tts_button_layout.addStretch()
        
        # Add the horizontal layout to the main vertical layout
        self.tts_layout.addLayout(self.tts_button_layout)

        self.tts_scroll = QScrollArea()
        self.tts_content = QTextEdit()
        self.tts_content.setReadOnly(False)
        self.tts_content.setText("Enter text here for Text-To-Speech...")
        self.tts_content.focusInEvent = lambda event: self.clear_placeholder_tts(event)
        self.tts_scroll.setWidgetResizable(True)
        self.tts_scroll.setWidget(self.tts_content)
        self.tts_layout.addWidget(self.tts_scroll)

        self.tabs.addTab(self.tts_tab, "Text To Speech")

        self.stt_tab = QWidget()
        self.stt_layout = QVBoxLayout(self.stt_tab)

        # Create horizontal layout for STT buttons
        self.stt_button_layout = QHBoxLayout()
        
        # Add Speech to Text button
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

        self.main_layout.addWidget(self.tabs)

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
        if dark:
            bg_color = "#333333"
            top_bar_color = "#008080"
            btn_color = "#00b3b3"
            hover_color = "#66ffff"
            transcription_bg = "#008080"
            text_color = "white"
        else:
            bg_color = "#f5f5f5"
            top_bar_color = "#00b3b3"
            btn_color = "#008080"
            hover_color = "#00cccc"
            transcription_bg = "#00b3b3"
            text_color = "black"

        self.setStyleSheet(f"background-color: {bg_color};")
        self.top_bar.setStyleSheet(f"background-color: {top_bar_color};")
        self.logo_label.setStyleSheet(f"color: {text_color};")

        for btn in [
            self.user_button,
            self.text_to_speech_btn,
            self.download_audio_btn,
            self.speech_to_text_btn,
            self.mic_select_btn
        ]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {btn_color};
                    color: {text_color};
                    border-radius: 5px;
                    height: 30px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QPushButton:disabled {{
                    background-color: #666666;
                    color: #999999;
                }}
            """)

        self.camera_frame.setStyleSheet(f"background-color: {bg_color}; border-radius: 10px;")
        self.transcription_label.setStyleSheet(f"background-color: {transcription_bg}; color: {text_color}; padding: 5px;")
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")

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
        cleanup()  # Clean up TTS resources
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

    def handle_text_to_speech(self):
        """Convert text to speech and play it"""
        text = self.tts_content.toPlainText().strip()
        
        if not text or text == "Enter text here for Text-To-Speech...":
            QMessageBox.warning(self, "No Text", "Please enter text in the TTS area first.")
            return

        # Prepare preferences for TTS
        prefs_for_tts = {
            "translate_to": self.tts_translation,
            "voice": self.tts_voice,
            "speed": self.tts_speed
        }

        # Disable button while processing
        self.text_to_speech_btn.setEnabled(False)
        self.text_to_speech_btn.setText("Processing...")
        self.download_audio_btn.setEnabled(False)

        # Create and start worker thread
        self.tts_worker = TTSWorker(text, prefs_for_tts)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.error.connect(self.on_tts_error)
        self.tts_worker.start()

    def on_tts_finished(self, translated_text):
        """Called when TTS conversion and playback complete"""
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("Text to Speech")
        self.download_audio_btn.setEnabled(True)
        
        original_text = self.tts_content.toPlainText().strip()
        
        # Update TTS text box with translated text if translation was used
        if translated_text != original_text and self.tts_translation != "No Translation":
            self.tts_content.setText(translated_text)
        
        # Simple success message without showing the translated text
        QMessageBox.information(self, "Playback Complete", "Audio playback finished successfully!")

    def on_tts_error(self, error_message):
        """Called when TTS conversion or playback fails"""
        self.text_to_speech_btn.setEnabled(True)
        self.text_to_speech_btn.setText("Text to Speech")
        QMessageBox.critical(self, "TTS Error", f"Text-to-Speech failed:\n{error_message}")

    def handle_download_audio(self):
        """Download generated audio files to a selected folder"""
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
    
    def stop_stt_recording(self):
        """Stop STT recording and reset button"""
        if self.stt_worker:
            self.stt_worker.stop_recording()
            self.stt_worker.wait()  # Wait for thread to finish
            
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
        
        self.stt_content.append("\n⏹️ Recording stopped.\n")
    
    def on_stt_text_recognized(self, text):
        """Handle recognized text from STT"""
        self.stt_content.append(f"📝 {text}\n")
    
    def on_stt_status_update(self, status):
        """Handle status updates from STT"""
        self.stt_content.append(f"{status}\n")
    
    def on_stt_error(self, error):
        """Handle errors from STT"""
        self.stt_content.append(f"❌ Error: {error}\n")
        QMessageBox.critical(self, "Speech Recognition Error", error)
        self.stop_stt_recording()
    
    def show_microphone_selection(self):
        """Show microphone selection dialog"""
        try:
            # Get list of available microphones
            mic_names = sr.Microphone.list_microphone_names()
            
            if not mic_names:
                QMessageBox.warning(self, "No Microphones", "No microphones found on this system.")
                return
            
            # Create selection dialog
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Microphone")
            dialog.setFixedSize(400, 300)
            
            layout = QVBoxLayout(dialog)
            
            # Add label
            label = QLabel("Select your microphone:")
            layout.addWidget(label)
            
            # Add microphone list
            mic_list = QListWidget()
            for i, name in enumerate(mic_names):
                mic_list.addItem(f"{i}: {name}")
                # Select currently selected microphone
                if i == self.mic_device_index:
                    mic_list.setCurrentRow(i)
            layout.addWidget(mic_list)
            
            # Add buttons
            button_layout = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            # Apply dark mode styling
            if self.dark_mode:
                dialog.setStyleSheet(f"""
                    QDialog {{ background-color: #333333; color: white; }}
                    QLabel {{ color: white; }}
                    QListWidget {{ 
                        background-color: #222222; 
                        color: white; 
                        border: 1px solid #555555; 
                    }}
                    QPushButton {{
                        background-color: #008080;
                        color: white;
                        border-radius: 5px;
                        height: 30px;
                        padding: 5px;
                    }}
                    QPushButton:hover {{
                        background-color: #00b3b3;
                    }}
                """)
            
            # Show dialog and get result
            if dialog.exec() == QDialog.Accepted:
                selected_row = mic_list.currentRow()
                if selected_row >= 0:
                    self.mic_device_index = selected_row
                    selected_name = mic_names[selected_row]
                    QMessageBox.information(
                        self, 
                        "Microphone Selected", 
                        f"Selected microphone:\n{selected_name}"
                    )
                    
                    # Save to user preferences
                    data = load_user_data()
                    if self.username not in data:
                        data[self.username] = {"preferences": {}}
                    if "preferences" not in data[self.username]:
                        data[self.username]["preferences"] = {}
                    
                    data[self.username]["preferences"]["mic_device_index"] = self.mic_device_index
                    data[self.username]["preferences"]["mic_device_name"] = selected_name
                    save_user_data(data)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get microphone list:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication([])

    # Show login flow (handles both login and signup)
    username = show_login_flow()
    
    if username:
        window = EchoMeApp(username)
        window.show()
        app.exec()
    else:
        # User cancelled login/signup
        print("Login cancelled by user")