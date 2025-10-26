"""
Preferences Dialog Module
Handles user preferences dialog and settings
"""

from PySide6.QtWidgets import QDialog, QFormLayout, QCheckBox, QComboBox, QPushButton, QLabel
from PySide6.QtCore import Qt
from tts import LANGUAGE_OPTIONS, GTTS_VOICES
from user_data import load_user_data, save_user_data


class PreferencesDialog(QDialog):
    """User preferences dialog window"""
    
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

        save_btn = QPushButton("Save")
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
        """Apply styling based on dark/light mode"""
        # Color palette configuration
        if dark_mode:
            # Dark mode colors
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
            # Light mode colors
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
            
            QFrame {{
                background-color: {card_bg};
                border-radius: 12px;
                padding: 15px;
                border: 1px solid {border_color};
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
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {text_color};
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {card_bg};
                color: {text_color};
                border: 2px solid {accent};
                border-radius: 8px;
                selection-background-color: {accent};
                selection-color: white;
                padding: 5px;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border-radius: 4px;
                margin: 2px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {accent_glow};
                color: {text_color};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {accent};
                color: white;
            }}
        """)

    def save_preferences(self):
        """Save user preferences"""
        if self.parent_window and self.user:
            # Update parent window settings
            if hasattr(self.parent_window, 'landmark_toggle_btn_state'):
                self.parent_window.landmark_toggle_btn_state(self.show_landmarks_checkbox.isChecked())
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