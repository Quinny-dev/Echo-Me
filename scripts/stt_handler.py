"""
Speech-to-Text Handler Module
Handles STT worker threads and STT functionality
"""

import speech_recognition as sr
import queue
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PySide6.QtCore import Qt
from user_data import load_user_data, save_user_data
from custom_popups import show_error, show_warning, show_info, StyledDialog
from styling import ThemeManager


class STTWorker(QThread):
    """Worker thread for STT processing"""
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


class STTHandler:
    """Handles STT functionality for the GUI"""
    
    def __init__(self, parent_window, username):
        self.parent = parent_window
        self.username = username
        self.stt_worker = None
        self.stt_is_recording = False
        self.mic_device_index = None
        self.theme_manager = ThemeManager()
    
    def update_mic_device(self, device_index):
        """Update microphone device index"""
        self.mic_device_index = device_index
    
    def handle_speech_to_text(self, stt_content, speech_to_text_btn):
        """Start or stop speech-to-text recording"""
        if not self.stt_is_recording:
            # Start recording
            stt_content.clear()
            stt_content.append("🎤 Starting speech recognition...\n")
            
            speech_to_text_btn.setText("⏹ Stop Recording")
            speech_to_text_btn.setStyleSheet("""
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
            self.stt_worker.text_recognized.connect(lambda text: self.on_stt_text_recognized(text, stt_content))
            self.stt_worker.status_update.connect(lambda status: self.on_stt_status_update(status, stt_content))
            self.stt_worker.error.connect(lambda error: self.on_stt_error(error, speech_to_text_btn))
            self.stt_worker.start()
            
        else:
            # Stop recording
            self.stop_stt_recording(speech_to_text_btn)

    def stop_stt_recording(self, speech_to_text_btn, dark_mode=True):
        """Stop STT recording and reset button"""
        if self.stt_worker:
            self.stt_worker.stop_recording()
            self.stt_worker.wait()
            
        self.stt_is_recording = False
        speech_to_text_btn.setText("🎤 Speech to Text")
        
        # Reset button style based on theme
        if dark_mode:
            btn_color = "#00b3b3"
            hover_color = "#66ffff"
        else:
            btn_color = "#008080"
            hover_color = "#00cccc"
            
        speech_to_text_btn.setStyleSheet(f"""
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

    def on_stt_text_recognized(self, text, stt_content):
        """Handle recognized text from STT"""
        current_text = stt_content.toPlainText()
        if "🎤 Starting speech recognition..." in current_text:
            stt_content.clear()
        
        if not text.startswith("⚠️") and not text.startswith("API error"):
            stt_content.append(f"{text}")
        else:
            stt_content.append(f"<i>{text}</i>")

    def on_stt_status_update(self, status, stt_content):
        """Handle STT status updates"""
        current_text = stt_content.toPlainText()
        if "🎤 Starting speech recognition..." in current_text:
            stt_content.clear()
        stt_content.append(f"<i>{status}</i>")

    def on_stt_error(self, error_message, speech_to_text_btn):
        """Handle STT errors"""
        self.stop_stt_recording(speech_to_text_btn)
        show_error(self.parent, "Speech Recognition Error", f"Speech-to-Text failed:\n{error_message}", self.parent.dark_mode)

    def show_microphone_selection(self):
        """Show microphone selection dialog"""
        try:
            import pyaudio
            
            # Get list of all devices
            all_devices = sr.Microphone.list_microphone_names()
            
            # Filter for input devices only using PyAudio
            p = pyaudio.PyAudio()
            input_devices = []
            seen_names = set()
            
            for i in range(len(all_devices)):
                try:
                    device_info = p.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels', 0) > 0:
                        device_name = all_devices[i]
                        # Skip if we've already seen this device name
                        if device_name not in seen_names:
                            seen_names.add(device_name)
                            input_devices.append({
                                'index': i,
                                'name': device_name,
                                'channels': device_info.get('maxInputChannels')
                            })
                except:
                    continue
            
            p.terminate()
            
            if not input_devices:
                show_warning(self.parent, "No Microphones", "No input devices found on this system.", self.parent.dark_mode)
                return
            
            # Create styled dialog
            dialog = StyledDialog(self.parent, "Select Microphone", self.parent.dark_mode)
            dialog.setFixedSize(420, 320)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Header
            header_label = QLabel("🎧 Select Input Device")
            header_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #3b82f6; margin-bottom: 10px;")
            layout.addWidget(header_label)
            
            label = QLabel("Choose your preferred microphone from the available input devices:")
            layout.addWidget(label)
            
            # Add microphone list with only input devices
            mic_combo = QComboBox()
            device_indices = []
            
            for device in input_devices:
                mic_combo.addItem(f"🎤 {device['name']}")
                device_indices.append(device['index'])
            
            # Select currently selected microphone if it's in the list
            if self.mic_device_index is not None and self.mic_device_index in device_indices:
                combo_index = device_indices.index(self.mic_device_index)
                mic_combo.setCurrentIndex(combo_index)
            
            layout.addWidget(mic_combo)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setFixedSize(80, 32)
            ok_btn = QPushButton("Select")
            ok_btn.setFixedSize(80, 32)
            
            def on_ok():
                # Get the actual device index from our filtered list
                combo_index = mic_combo.currentIndex()
                if combo_index >= 0 and combo_index < len(device_indices):
                    self.mic_device_index = device_indices[combo_index]
                    selected_name = mic_combo.currentText().replace("🎤 ", "")
                else:
                    show_error(dialog, "Invalid Selection", "Please select a valid microphone device.", self.parent.dark_mode)
                    return
                
                # Save to user preferences
                data = load_user_data()
                if self.username not in data:
                    data[self.username] = {"preferences": {}}
                elif "preferences" not in data[self.username]:
                    data[self.username]["preferences"] = {}
                
                data[self.username]["preferences"]["mic_device_index"] = self.mic_device_index
                save_user_data(data)
                
                dialog.accept()
                show_info(
                    self.parent, 
                    "Microphone Selected", 
                    f"Successfully selected:\n{selected_name}",
                    True,
                    self.parent.dark_mode
                )
            
            def on_cancel():
                dialog.reject()
            
            ok_btn.clicked.connect(on_ok)
            cancel_btn.clicked.connect(on_cancel)
            
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(ok_btn)
            layout.addLayout(button_layout)
            
            # Apply styling
            colors = self.theme_manager.get_color_palette(self.parent.dark_mode)
            dialog.setStyleSheet(self.theme_manager.get_dialog_style(colors))
            
            dialog.exec()
            
        except Exception as e:
            show_error(self.parent, "Microphone Error", f"Failed to get microphone list:\n{str(e)}", self.parent.dark_mode)