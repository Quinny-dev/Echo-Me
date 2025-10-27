"""
Custom Popup System - Modern, Animated Popups with Glassmorphism
Provides consistent styling for all dialogs and message boxes in Echo Me
"""

from PySide6.QtWidgets import (
    QDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QWidget, QGraphicsOpacityEffect, QFrame
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QRect
from PySide6.QtGui import QFont, QPainter, QBrush, QColor, QPen, QPixmap, QIcon
from enum import Enum


class PopupType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class StyledMessageBox(QDialog):
    """Modern styled message box with animations and glassmorphism"""
    
    def __init__(self, parent=None, title="", message="", popup_type=PopupType.INFO, 
                 auto_close=False, auto_close_delay=3000, dark_mode=True):
        super().__init__(parent)
        self.popup_type = popup_type
        self.auto_close = False
        self.auto_close_delay = auto_close_delay
        self.dark_mode = dark_mode
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        
        self.setup_ui(title, message)
        self.apply_styling()
        self.setup_animations()
        
        if self.auto_close:
            QTimer.singleShot(self.auto_close_delay, self.close)
    
    def setup_ui(self, title, message):
        """Setup the UI structure"""
        # Main container with rounded corners
        self.container = QFrame(self)
        self.container.setObjectName("messageContainer")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
        
        # Container layout
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(20, 20, 20, 20)
        container_layout.setSpacing(15)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        # Icon
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(32, 32)
        self.icon_label.setScaledContents(True)
        self.set_icon()
        header_layout.addWidget(self.icon_label)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        
        container_layout.addLayout(header_layout)
        
        # Message
        self.message_label = QLabel(message)
        self.message_label.setObjectName("messageLabel")
        self.message_label.setWordWrap(True)
        container_layout.addWidget(self.message_label)
        
        # Button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.setObjectName("okButton")
        self.ok_button.setFixedSize(80, 32)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        container_layout.addLayout(button_layout)
        
        # Set size
        self.setFixedSize(400, 180)
    
    def set_icon(self):
        """Set icon based on popup type"""
        icons = {
            PopupType.SUCCESS: "✓",
            PopupType.ERROR: "✕",
            PopupType.WARNING: "⚠",
            PopupType.INFO: "ℹ"
        }
        
        colors = {
            PopupType.SUCCESS: "#10b981",
            PopupType.ERROR: "#ef4444",
            PopupType.WARNING: "#f59e0b",
            PopupType.INFO: "#3b82f6"
        }
        
        # Create colored icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle background
        painter.setBrush(QBrush(QColor(colors[self.popup_type])))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 32, 32)
        
        # Draw icon text
        painter.setPen(QPen(Qt.white, 2))
        font = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font)
        painter.drawText(QRect(0, 0, 32, 32), Qt.AlignCenter, icons[self.popup_type])
        
        painter.end()
        
        self.icon_label.setPixmap(pixmap)
    
    def apply_styling(self):
        """Apply modern styling with glassmorphism effect"""
        # Colors based on theme
        if self.dark_mode:
            bg_color = "rgba(26, 26, 26, 0.95)"
            border_color = "#2d3748"
            text_color = "#f0f0f0"
            button_bg = "#004D61"
            button_hover = "#27A2A7"
        else:
            bg_color = "rgba(255, 255, 255, 0.95)"
            border_color = "#e2e8f0"
            text_color = "#111925"
            button_bg = "#79BDBD"
            button_hover = "#74A5A5"
        
        # Accent colors based on popup type
        accent_colors = {
            PopupType.SUCCESS: "#10b981",
            PopupType.ERROR: "#ef4444",
            PopupType.WARNING: "#f59e0b",
            PopupType.INFO: "#3b82f6"
        }
        
        accent = accent_colors[self.popup_type]
        
        style = f"""
            #messageContainer {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 16px;
                border-top: 3px solid {accent};
            }}
            
            #titleLabel {{
                color: {text_color};
                font-size: 16px;
                font-weight: 700;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            #messageLabel {{
                color: {text_color};
                font-size: 14px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                line-height: 1.5;
                padding: 5px 0;
            }}
            
            #okButton {{
                background-color: {button_bg};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            #okButton:hover {{
                background-color: {button_hover};
            }}
            
            #okButton:pressed {{
                background-color: {accent};
            }}
        """
        
        self.setStyleSheet(style)
    
    def setup_animations(self):
        """Setup fade and scale animations"""
        # Opacity animation
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(200)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # Start animation
        self.fade_animation.start()
    
    def closeEvent(self, event):
        """Animate closing"""
        self.fade_animation.setDirection(QPropertyAnimation.Backward)
        self.fade_animation.finished.connect(lambda: super().closeEvent(event))
        self.fade_animation.start()
        event.ignore()


class StyledDialog(QDialog):
    """Base class for styled dialogs with consistent theming"""
    
    def __init__(self, parent=None, title="", dark_mode=True):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)
        
    def apply_dialog_styling(self):
        """Apply consistent dialog styling"""
        if self.dark_mode:
            bg_color = "#1A1A1A"
            panel_bg = "#131720"
            text_color = "#F0F0F0"
            border_color = "#1e293b"
            button_bg = "#004D61"
            button_hover = "#27A2A7"
            input_bg = "#1a1f2e"
        else:
            bg_color = "#FFFFFF"
            panel_bg = "#f8fafc"
            text_color = "#111925"
            border_color = "#e2e8f0"
            button_bg = "#79BDBD"
            button_hover = "#74A5A5"
            input_bg = "#ffffff"
        
        return f"""
            QDialog {{
                background-color: {bg_color};
                color: {text_color};
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            QLabel {{
                color: {text_color};
                font-size: 14px;
                padding: 5px 0;
            }}
            
            QComboBox {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                min-height: 20px;
            }}
            
            QComboBox:hover {{
                border-color: {button_bg};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {text_color};
                width: 0;
                height: 0;
                margin-right: 5px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {panel_bg};
                color: {text_color};
                selection-background-color: {button_bg};
                selection-color: white;
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 5px;
            }}
            
            QPushButton {{
                background-color: {button_bg};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
            }}
            
            QPushButton:hover {{
                background-color: {button_hover};
            }}
            
            QPushButton:pressed {{
                background-color: {button_bg};
                padding-top: 9px;
                padding-bottom: 7px;
            }}
            
            QFrame {{
                background-color: {panel_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
                padding: 20px;
            }}
        """


# Convenience functions for quick popup creation
def show_success(parent, title, message, auto_close=False, dark_mode=True):
    """Show a success popup"""
    popup = StyledMessageBox(parent, title, message, PopupType.SUCCESS, auto_close, dark_mode=dark_mode)
    popup.exec()


def show_error(parent, title, message, dark_mode=True):
    """Show an error popup"""
    popup = StyledMessageBox(parent, title, message, PopupType.ERROR, False, dark_mode=dark_mode)
    popup.exec()


def show_warning(parent, title, message, dark_mode=True):
    """Show a warning popup"""
    popup = StyledMessageBox(parent, title, message, PopupType.WARNING, False, dark_mode=dark_mode)
    popup.exec()


def show_info(parent, title, message, auto_close=False, dark_mode=True):
    """Show an info popup"""
    popup = StyledMessageBox(parent, title, message, PopupType.INFO, auto_close, dark_mode=dark_mode)
    popup.exec()