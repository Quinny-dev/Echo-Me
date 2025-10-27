"""
Styling and Themes Module - Enhanced Visual Design
Premium color palettes and refined typography for better aesthetics
"""


class ThemeManager:
    """Manages application themes and styling with enhanced visual design"""
    
    def __init__(self):
        self.current_theme = "dark"
    
    def get_color_palette(self, dark_mode=True):
        """Get enhanced color palette for the specified theme"""
        if dark_mode:
            return {
                # Dark mode: Modern deep blue theme with vibrant accents
                'bg_color': "#0d1117",              # Main dialog background
                'card_bg': "#161b22",               # Card/panel backgrounds
                'top_bar_color': "#161b22",         # Top bar base color (gradient start)
                'top_bar_gradient': "#1f6feb",      # Top bar gradient accent (gradient end)
                'btn_primary': "#1f6feb",           # Primary button color (accent)
                'btn_hover': "#58a6ff",             # Button hover state (lighter accent)
                'btn_glow': "rgba(88, 166, 255, 0.4)",  # Glow effect color for buttons
                'transcription_bg': "#161b22",      # Transcription header background
                'text_color': "#e6edf3",            # Primary text color
                'text_color_bright': "#f3f3f3",     # Extra bright for emphasis
                'camera_bg': "#161b22",             # Camera panel background
                'menu_bg': "#161b22",               # Menu panel background
                'scroll_bg': "#161b22",             # Scroll container background
                'text_edit_bg': "#0d1117",          # Text editor background
                'border_color': "#30363d",          # Subtle borders
                'shadow_dark': "rgba(0, 0, 0, 0.6)", # Strong shadow for depth
                'shadow_light': "rgba(0, 0, 0, 0.3)",# Lighter shadow
                'disabled_bg': "#1e293b",           # Disabled control background
                'disabled_text': "#64748b",         # Disabled text color
                'accent_purple': "#8b5cf6",         # Optional purple accent
                'accent_cyan': "#06b6d4",           # Optional cyan accent
                'accent': "#1f6feb",
                'accent_hover': "#58a6ff",
                'accent_glow': "rgba(88, 166, 255, 0.4)",
                'checkbox_bg': "#161b22",
                'combo_bg': "#0d1117",
                'shadow_color': "rgba(0, 0, 0, 0.6)"
            }
        else:
            return {
                # Light mode: Clean, modern palette with warm undertones
                'bg_color': "#E8E8E8",                      # Main window background
                'card_bg': "#ffffff",                       # Card/panel backgrounds
                'top_bar_color': "#B384FF",                 # Top bar gradient START (light)
                'top_bar_gradient': "#59B6F8",              # Top bar gradient END (darker) -> creates light->dark gradient
                'btn_primary': "#727AFF",                   # Primary button color
                'btn_hover': "#492077",                     # Button hover state
                'btn_glow': "rgba(203, 243, 187, 0.5)",     # Button glow effect
                'transcription_bg': "#ffffff",              # Transcription header background
                'text_color': "#000000",                    # Primary text color
                'text_color_bright': "#FFFFFF",             # Extra bright (for dark text on accent)
                'camera_bg': "#ffffff",                     # Camera panel background
                'menu_bg': "#ffffff",                       # Menu panel background
                'scroll_bg': "#ffffff",                     # Scroll container background
                'text_edit_bg': "#FFFFFF",                  # Text editor background
                'border_color': "#e2e8f0",                  # Border color
                'shadow_dark': "rgba(0, 0, 0, 0.9)",        # Darker shadow for light theme
                'shadow_light': "rgba(0, 0, 0, 0.01)",      # Lighter shadow
                'disabled_bg': "#cbd5e1",                   # Disabled control background
                'disabled_text': "#64748b",                 # Disabled text color
                'accent_purple': "#7c3aed",                 # Optional purple accent
                'accent_cyan': "#0891b2",                   # Optional cyan accent
                'accent': "#79BDBD",
                'accent_hover': "#74A5A5",
                'accent_glow': "rgba(203, 243, 187, 0.5)",
                'checkbox_bg': "#ffffff",
                'combo_bg': "#f6f8fa",
                'shadow_color': "rgba(0, 0, 0, 0.08)"
            }

    
    def get_button_style(self, colors):
        """Get enhanced button styling with modern gradients"""
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {colors['btn_primary']}, 
                    stop:0.5 {colors['btn_primary']}, 
                    stop:1 {colors['btn_hover']});
                color: white;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
                border: none;
                letter-spacing: 0.5px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                padding: 2px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {colors['btn_hover']}, 
                    stop:1 {colors['btn_primary']});
                box-shadow: 0 4px 16px {colors['btn_glow']};
            }}
            QPushButton:pressed {{
                background: {colors['btn_hover']};
                padding-top: 3px;
            }}
            QPushButton:disabled {{
                background: {colors['disabled_bg']};
                color: {colors['disabled_text']};
            }}
        """
    
    def get_scroll_area_style(self, colors):
        """Get enhanced scroll area styling"""
        return f"""
            QScrollArea {{
                background-color: {colors['scroll_bg']};
                border: 1px solid {colors['border_color']};
                border-radius: 12px;
            }}
            
            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                border-radius: 6px;
                margin: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['btn_primary']};
                border-radius: 6px;
                min-height: 30px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {colors['btn_hover']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
    
    def get_text_edit_style(self, colors):
        """Get enhanced text editor styling with better typography"""
        return f"""
            QTextEdit {{
                background-color: {colors['text_edit_bg']};
                color: {colors['text_color_bright']};
                border: none;
                border-radius: 10px;
                padding: 18px;
                font-size: 13px;
                line-height: 1.7;
                font-weight: 400;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                selection-background-color: {colors['btn_primary']};
                selection-color: white;
                letter-spacing: 0.3px;
            }}
        """
    
    def get_tab_widget_style(self, colors):
        """Get enhanced tab widget styling"""
        return f"""
            QTabWidget::pane {{
                background-color: {colors['scroll_bg']};
                border: 1px solid {colors['border_color']};
                border-radius: 12px;
            }}
            QTabBar::tab {{
                background-color: transparent;
                color: {colors['text_color']};
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: none;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                letter-spacing: 0.3px;
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
        """Get enhanced main window styling"""
        return f"""
            QWidget {{
                background-color: {colors['bg_color']};
                color: {colors['text_color']};
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
        """
    
    def get_top_bar_style(self, colors):
        """Get enhanced top bar styling with modern gradient"""
        return f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 {colors['top_bar_color']}, 
                    stop:0.5 {colors['top_bar_gradient']},
                    stop:1 {colors['top_bar_color']});
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 15px;  
                border-bottom-right-radius: 15px;
                border: 1px solid {colors['border_color']};
            }}
            QLabel {{
                color: {colors['text_color_bright']};
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 3px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                text-shadow: 2px 2px 4px {colors['shadow_dark']};
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 15px;  
                border-bottom-right-radius: 15px;
            }}
            QPushButton {{
                color: {colors['text_color_bright']};
                border-radius: 18px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
        """
    
    def get_camera_frame_style(self, colors):
        """Get enhanced camera frame styling"""
        return f"""
            QFrame {{
                background-color: {colors['camera_bg']}; 
                border-radius: 18px;
                border: 1px solid {colors['border_color']};
            }}
        """
    
    def get_camera_label_style(self, colors):
        """Get enhanced camera label styling"""
        return f"""
            QLabel {{
                background-color: #000000;
                border-radius: 14px;
                border: 2px solid {colors['border_color']};
            }}
        """
    
    def get_dialog_style(self, colors):
        """Get enhanced dialog styling for custom popups"""
        return f"""
            QDialog {{
                background-color: {colors['bg_color']};
                color: {colors['text_color']};
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
                border-radius: 16px;
            }}
            
            QLabel {{
                color: {colors['text_color']};
                font-size: 14px;
                padding: 5px 0;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            QComboBox {{
                background-color: {colors['text_edit_bg']};
                color: {colors['text_color']};
                border: 1px solid {colors['border_color']};
                border-radius: 8px;
                padding: 10px 35px 10px 12px;
                font-size: 13px;
                min-height: 20px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            QComboBox:hover {{
                border-color: {colors['btn_primary']};
            }}
            
            QComboBox::down-arrow {{
                subcontrol-origin: padding;
                subcontrol-position: center;
                width: 16px;
                height: 16px;
                font-size: 12px;
                
                font-weight: bold;
            }}
            
            QComboBox::down-arrow:after {{
                content: "▼";
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
            }}
            
            
            QComboBox QAbstractItemView {{
                background-color: {colors['menu_bg']};
                color: {colors['text_color']};
                selection-background-color: {colors['btn_primary']};
                selection-color: white;
                border: 1px solid {colors['border_color']};
                border-radius: 8px;
                padding: 5px;
            }}
            
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {colors['btn_primary']}, 
                    stop:1 {colors['btn_hover']});
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {colors['btn_hover']}, 
                    stop:1 {colors['btn_primary']});
                box-shadow: 0 4px 16px {colors['btn_glow']};
            }}
            
            QPushButton:pressed {{
                background: {colors['btn_hover']};
                padding-top: 9px;
                padding-bottom: 7px;
            }}
        """


def apply_theme(widget, theme_manager, dark_mode=True):
    """Apply enhanced theme to a widget and its children"""
    colors = theme_manager.get_color_palette(dark_mode)
    widget.setStyleSheet(theme_manager.get_main_window_style(colors))
    return colors