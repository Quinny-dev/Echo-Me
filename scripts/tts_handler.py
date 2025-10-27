"""
Text-to-Speech Handler Module
Handles TTS worker threads and TTS functionality
"""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QFileDialog
from pathlib import Path
from tts import convert_and_play, download_audio_files, play_word_instantly
import threading

class TTSWorker(QThread):
    """Worker thread for TTS processing"""
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


class TTSHandler:
    """Handles TTS functionality for the GUI"""

    def __init__(self, parent_window):
        self.parent = parent_window
        self.tts_worker = None

        # TTS settings
        self.tts_translation = "No Translation"
        self.tts_voice = "English (US)"
        self.tts_speed = "Normal"
        
        print("✅ TTSHandler initialized")

    def update_settings(self, translation=None, voice=None, speed=None):
        """Update TTS settings"""
        if translation is not None:
            self.tts_translation = translation
            print(f"🔧 TTS Translation updated: {translation}")
        if voice is not None:
            self.tts_voice = voice
            print(f"🔧 TTS Voice updated: {voice}")
        if speed is not None:
            self.tts_speed = speed
            print(f"🔧 TTS Speed updated: {speed}")

    def handle_text_to_speech(self, tts_content, text_to_speech_btn, download_audio_btn):
        """Handle TTS button click"""
        text = tts_content.toPlainText().strip()

        if not text or text == "Model predictions will appear here...":
            QMessageBox.warning(self.parent, "No Text", "Please enter text in the TTS area first.")
            return

        prefs_for_tts = {
            "translate_to": self.tts_translation,
            "voice": self.tts_voice,
            "speed": self.tts_speed
        }

        text_to_speech_btn.setEnabled(False)
        text_to_speech_btn.setText("Processing...")
        download_audio_btn.setEnabled(False)

        self.tts_worker = TTSWorker(text, prefs_for_tts)
        self.tts_worker.finished.connect(
            lambda translated_text: self.on_tts_finished(
                translated_text, tts_content, text_to_speech_btn, download_audio_btn
            )
        )
        self.tts_worker.error.connect(
            lambda error: self.on_tts_error(
                error, text_to_speech_btn, download_audio_btn
            )
        )
        self.tts_worker.start()

    def handle_text_to_speech_live(self, text):
        """Play TTS immediately when a sign is detected (non-blocking)."""
        print(f"🔊 LIVE TTS CALLED with text: '{text}'")
        
        prefs_for_tts = {
            "translate_to": self.tts_translation,
            "voice": self.tts_voice,
            "speed": self.tts_speed
        }
        
        print(f"📋 TTS Preferences: {prefs_for_tts}")
        
        try:
            # Test if play_word_instantly is callable
            print(f"🔍 play_word_instantly function: {play_word_instantly}")
            
            # Start thread
            thread = threading.Thread(
                target=play_word_instantly,
                args=(text, prefs_for_tts),
                daemon=True
            )
            print(f"🧵 Thread created: {thread}")
            thread.start()
            print(f"✅ Thread started successfully")
            
        except Exception as e:
            print(f"❌ Live TTS failed with exception: {e}")
            import traceback
            traceback.print_exc()

    def on_tts_finished(self, translated_text, tts_content, text_to_speech_btn, download_audio_btn):
        """Handle TTS completion"""
        text_to_speech_btn.setEnabled(True)
        text_to_speech_btn.setText("🔊 Text to Speech")
        download_audio_btn.setEnabled(True)

        original_text = tts_content.toPlainText().strip()
        if translated_text != original_text and self.tts_translation != "No Translation":
            tts_content.setText(translated_text)

        QMessageBox.information(self.parent, "Playback Complete", "Audio playback finished successfully!")

    def on_tts_error(self, error_message, text_to_speech_btn, download_audio_btn):
        """Handle TTS error"""
        text_to_speech_btn.setEnabled(True)
        text_to_speech_btn.setText("🔊 Text to Speech")
        download_audio_btn.setEnabled(False)
        QMessageBox.critical(self.parent, "TTS Error", f"Text-to-Speech failed:\n{error_message}")

    def handle_download_audio(self):
        """Handle audio download"""
        folder_selected = QFileDialog.getExistingDirectory(
            self.parent,
            "Select Download Folder",
            str(Path.home() / "Downloads")
        )

        if not folder_selected:
            return

        try:
            num_files = download_audio_files(folder_selected)
            QMessageBox.information(
                self.parent,
                "Download Complete",
                f"Successfully saved {num_files} audio file(s) to:\n{folder_selected}"
            )
        except Exception as e:
            QMessageBox.critical(self.parent, "Download Error", f"Failed to download audio files:\n{str(e)}")