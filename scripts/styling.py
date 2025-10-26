"""
Styling and Themes Module - Fixed Color Contrast
Handles all UI styling and theme management with improved visibility
"""


class ThemeManager:
    """Manages application themes and styling"""
    
    def __init__(self):
        self.current_theme = "dark"
    
    def get_color_palette(self, dark_mode=True):
        """Get color palette for the specified theme"""
        if dark_mode:
            return {
                # Dark mode: GitHub-inspired dark theme with blue accents - IMPROVED CONTRAST
                'bg_color': "#0d1117",                      # Main window background
                'top_bar_color': "#161b22",                 # Top bar base color
                'top_bar_gradient': "#1c2938",              # Darker gradient (less blue, more readable)
                'btn_primary': "#58a6ff",                   # Brighter primary button
                'btn_hover': "#79b8ff",                     # Lighter hover state
                'btn_glow': "rgba(88, 166, 255, 0.4)",      # Button glow effect
                'transcription_bg': "#161b22",              # Transcription label background
                'text_color': "#c9d1d9",                    # Main text color (better contrast)
                'text_color_bright': "#e6edf3",             # Bright text for emphasis
                'camera_bg': "#161b22",                     # Camera frame background
                'menu_bg': "#161b22",                       # Menu panel background
                'scroll_bg': "#161b22",                     # Scroll area background
                'text_edit_bg': "#0d1117",                  # Text editor background
                'border_color': "#30363d",                  # Border color for all elements
                'shadow_dark': "rgba(0, 0, 0, 0.6)",        # Strong shadow for elevation
                'shadow_light': "rgba(0, 0, 0, 0.3)",       # Light shadow for subtle depth
                'disabled_bg': "#21262d",                   # Disabled button background
                'disabled_text': "#8b949e"                  # Disabled button text
            }
        else:
            return {
                # Light mode: Clean, professional palette with blue accents
                'bg_color': "#f6f8fa",                      # Main window background
                'top_bar_color': "#ffffff",                 # Top bar base color
                'top_bar_gradient': "#e8f0fe",              # Light blue gradient
                'btn_primary': "#0969da",                   # Primary button color
                'btn_hover': "#0550ae",                     # Button hover state
                'btn_glow': "rgba(9, 105, 218, 0.3)",       # Button glow effect
                'transcription_bg': "#ffffff",              # Transcription label background
                'text_color': "#24292f",                    # Main text color
                'text_color_bright': "#24292f",             # Same in light mode
                'camera_bg': "#ffffff",                     # Camera frame background
                'menu_bg': "#ffffff",                       # Menu panel background
                'scroll_bg': "#ffffff",                     # Scroll area background
                'text_edit_bg': "#f6f8fa",                  # Text editor background
                'border_color': "#d0d7de",                  # Border color for all elements
                'shadow_dark': "rgba(0, 0, 0, 0.15)",       # Strong shadow for elevation
                'shadow_light': "rgba(0, 0, 0, 0.08)",      # Light shadow for subtle depth
                'disabled_bg': "#94a3b8",                   # Disabled button background
                'disabled_text': "#64748b"                  # Disabled button text
            }
    
    def get_button_style(self, colors):
        """Get universal button styling"""
        return f"""
            QPushButton {{
                /* Vertical gradient from primary to hover color */
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {colors['btn_primary']}, stop:1 {colors['btn_hover']});
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
                    stop:0 {colors['btn_hover']}, stop:1 {colors['btn_primary']});
                box-shadow: 0 4px 12px {colors['btn_glow']};  /* Elevated shadow effect */
            }}
            /* Pressed/clicked state */
            QPushButton:pressed {{
                background: {colors['btn_hover']};
                padding-top: 1px;          /* Subtle downward press animation */
            }}
            /* Disabled state (e.g., Download Audio when inactive) */
            QPushButton:disabled {{
                background: {colors['disabled_bg']};
                color: {colors['disabled_text']};  /* Muted text for disabled state */
            }}
        """
    
    def get_scroll_area_style(self, colors):
        """Get scroll area styling"""
        return f"""
            QScrollArea {{
                background-color: {colors['scroll_bg']};
                border: 1px solid {colors['border_color']};
                border-radius: 12px;
            }}
            
            /* ---- Custom Scrollbar ---- */
            /* Vertical scrollbar track */
            QScrollBar:vertical {{
                background: {colors['text_edit_bg']};
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }}
            /* Scrollbar handle (the draggable part) */
            QScrollBar::handle:vertical {{
                background: {colors['btn_primary']};
                border-radius: 5px;
                min-height: 20px;
                margin: 2px;
            }}
            /* Scrollbar handle hover state */
            QScrollBar::handle:vertical:hover {{
                background: {colors['btn_hover']};
            }}
            /* Hide scrollbar arrow buttons */
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
    
    def get_text_edit_style(self, colors):
        """Get text editor styling"""
        return f"""
            QTextEdit {{
                background-color: {colors['text_edit_bg']};
                color: {colors['text_color_bright']};  /* Use bright text for readability */
                border: none;
                border-radius: 10px;
                padding: 16px;
                font-size: 11pt;
                line-height: 1.6;          /* Comfortable line spacing */
                font-weight: 400;          /* Regular weight for body text */
                selection-background-color: {colors['btn_primary']};  /* Highlighted text background */
                selection-color: white;                      /* Highlighted text color */
            }}
        """
    
    def get_tab_widget_style(self, colors):
        """Get tab widget styling"""
        return f"""
            QTabWidget::pane {{
                background-color: {colors['scroll_bg']};
                border: 1px solid {colors['border_color']};
                border-radius: 12px;
            }}
            QTabBar::tab {{
                background-color: {colors['camera_bg']};
                color: {colors['text_color']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {colors['border_color']};
            }}
            QTabBar::tab:selected {{
                background-color: {colors['btn_primary']};
                color: white;
            }}
            QTabBar::tab:hover {{
                background-color: {colors['btn_hover']};
                color: white;
            }}
        """
    
    def get_main_window_style(self, colors):
        """Get main window styling"""
        return f"""
            QWidget {{
                background-color: {colors['bg_color']};
                color: {colors['text_color']};
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            }}
        """
    
    def get_top_bar_style(self, colors):
        """Get top bar styling with better text visibility"""
        return f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {colors['top_bar_color']}, stop:1 {colors['top_bar_gradient']});
                border-radius: 20px;
                border: 1px solid {colors['border_color']};
            }}
            QLabel {{
                color: {colors['text_color_bright']};  /* Use bright text instead of white */
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 2px;
                text-shadow: 2px 2px 4px {colors['shadow_dark']};
            }}
            QPushButton {{
                color: {colors['text_color_bright']};  /* Ensure button text is visible */
            }}
        """
    
    def get_camera_frame_style(self, colors):
        """Get camera frame styling"""
        return f"""
            QFrame {{
                background-color: {colors['camera_bg']}; 
                border-radius: 18px;
                border: 1px solid {colors['border_color']};
            }}
        """
    
    def get_camera_label_style(self, colors):
        """Get camera label styling"""
        return f"""
            QLabel {{
                background-color: #000000;  /* Black for video feed */
                border-radius: 14px;
                border: 2px solid {colors['border_color']};
            }}
        """


def apply_theme(widget, theme_manager, dark_mode=True):
    """Apply theme to a widget and its children"""
    colors = theme_manager.get_color_palette(dark_mode)
    
    # Apply main window styling
    widget.setStyleSheet(theme_manager.get_main_window_style(colors))
    
    return colors