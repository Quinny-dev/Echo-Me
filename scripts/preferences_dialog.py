"""
Preferences Dialog Module - Enhanced Visual Design
Premium styling for settings dialog with refined colors and typography
"""

from PySide6.QtWidgets import QDialog, QFormLayout, QCheckBox, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt
from tts import LANGUAGE_OPTIONS, GTTS_VOICES
from user_data import load_user_data, save_user_data


class PreferencesDialog(QDialog):
    """Enhanced user preferences dialog with premium styling"""
    
    def __init__(self, parent=None, user=None):
        super().__init__(parent)
        self.setWindowTitle("User Preferences")
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        self.setFixedSize(450, 400)

        self.parent_window = parent
        self.user = user
        user_data = load_user_data().get(user, {}).get("preferences", {})

        # Create UI elements
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

        save_btn = QPushButton("Save Preferences")
        save_btn.clicked.connect(self.save_preferences)

        # Layout
        layout = QFormLayout(self)
        layout.addRow(self.dark_mode_checkbox)
        layout.addRow(self.show_landmarks_checkbox)
        layout.addRow(QLabel("Translate to:"), self.translation_combo)
        layout.addRow(QLabel("Voice:"), self.voice_combo)
        layout.addRow(QLabel("Speed:"), self.speed_combo)
        layout.addRow(save_btn)

        self.apply_styles(self.dark_mode_checkbox.isChecked())

    def apply_styles(self, dark_mode):
        """Apply enhanced styling based on dark/light mode"""
        if dark_mode:
            # Dark mode: Modern deep blue theme
            bg_color = "#1A1A1A"
            card_bg = "#132440"
            text_color = "#F0F0F0"
            label_color = "#F0F0F0"
            accent = "#004D61"
            accent_hover = "#27A2A7"
            accent_glow = "rgba(21, 179, 146, 0.5)"
            checkbox_bg = "#1a1f2e"
            combo_bg = "#1a1f2e"
            combo_text = "#f1f5f9"
            border_color = "#1e293b"
            hover_bg = "#1e293b"
        else:
            # Light mode: Clean, warm palette
            bg_color = "#fafbfc"
            card_bg = "#ffffff"
            text_color = "#1e293b"
            label_color = "#0f172a"
            accent = "#2563eb"
            accent_hover = "#1d4ed8"
            accent_glow = "rgba(37, 99, 235, 0.3)"
            checkbox_bg = "#ffffff"
            combo_bg = "#f8fafc"
            combo_text = "#0f172a"
            border_color = "#e2e8f0"
            hover_bg = "#f1f5f9"

        self.setStyleSheet(f"""
            QDialog {{ 
                background-color: {bg_color}; 
                color: {text_color}; 
                border-radius: 20px; 
            }}
            
            QLabel {{ 
                color: {label_color}; 
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                font-weight: 600;
                letter-spacing: 0.3px;
                font-size: 13px;
            }}
            
            QFrame {{
                background-color: {card_bg};
                border-radius: 12px;
                padding: 16px;
                border: 1px solid {border_color};
            }}
            
            QCheckBox {{ 
                spacing: 12px;
                color: {label_color};
                font-size: 13px;
                font-weight: 500;
                padding: 10px;
                border-radius: 8px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                letter-spacing: 0.3px;
            }}
            QCheckBox:hover {{
                background-color: {hover_bg};
            }}
            
            QCheckBox::indicator {{
                width: 22px; 
                height: 22px; 
                border-radius: 6px;
                border: 2px solid {border_color}; 
                background-color: {checkbox_bg};
            }}
            QCheckBox::indicator:hover {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 4px {accent_glow};
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
                    stop:0 {accent}, 
                    stop:0.5 {accent}, 
                    stop:1 {accent_hover});
                color: white; 
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 14px 28px;
                border: none;
                letter-spacing: 0.5px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {accent_hover}, 
                    stop:1 {accent});
                box-shadow: 0 4px 16px {accent_glow};
            }}
            QPushButton:pressed {{
                background: {accent};
                padding-top: 15px;
                padding-bottom: 13px;
            }}
            
            QComboBox {{
                background-color: {combo_bg};
                color: {combo_text};
                border: 2px solid {border_color};
                border-radius: 10px;
                padding: 11px 16px;
                font-size: 13px;
                font-weight: 500;
                min-height: 28px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                letter-spacing: 0.3px;
            }}
            QComboBox:hover {{
                border: 2px solid {accent};
                background-color: {hover_bg};
                box-shadow: 0 0 0 4px {accent_glow};
            }}
            QComboBox:focus {{
                border: 2px solid {accent};
                box-shadow: 0 0 0 4px {accent_glow};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 12px;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {combo_text};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {card_bg};
                color: {combo_text};
                border: 2px solid {accent};
                border-radius: 10px;
                selection-background-color: {accent};
                selection-color: white;
                padding: 6px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 10px 14px;
                border-radius: 6px;
                margin: 2px;
                color: {combo_text};
                font-weight: 500;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {hover_bg};
                color: {combo_text};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {accent};
                color: white;
                font-weight: 600;
            }}
        """)

    def save_preferences(self):
        """Save user preferences"""
        if self.parent_window and self.user:
            # Update parent window settings
            if getattr(self.parent_window, 'camera_handler', None):
                try:
                    self.parent_window.camera_handler.set_landmark_visibility(self.show_landmarks_checkbox.isChecked())
                except Exception:
                    pass
            if hasattr(self.parent_window, 'set_dark_mode'):
                self.parent_window.set_dark_mode(self.dark_mode_checkbox.isChecked())
            
            # Update TTS settings
            if hasattr(self.parent_window, 'tts_handler'):
                self.parent_window.tts_handler.update_settings(
                    translation=self.translation_combo.currentText(),
                    voice=self.voice_combo.currentText(),
                    speed=self.speed_combo.currentText()
                )

            # Save to file
            data = load_user_data()
            if self.user not in data:
                data[self.user] = {"preferences": {}}
            
            data[self.user]["preferences"] = {
                "dark_mode": self.dark_mode_checkbox.isChecked(),
                "show_landmarks": self.show_landmarks_checkbox.isChecked(),
                "tts_translation": self.translation_combo.currentText(),
                "tts_voice": self.voice_combo.currentText(),
                "tts_speed": self.speed_combo.currentText()
            }
            save_user_data(data)
        
        self.accept()