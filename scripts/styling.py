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
                'bg_color': "#1A1A1A",                      # Deeper, richer background
                'top_bar_color': "#1A1A1A",                  # Subtle blue-gray
                'top_bar_gradient': "#01144B",               # Gentle gradient
                'btn_primary': "#004D61",                    # Vibrant blue primary
                'btn_hover': "#27A2A7",                     # Bright blue hover
                'btn_glow': "rgba(21, 179, 146, 0.5)",        #Stronger glow
                'transcription_bg': "#131720",              # Consistent with top bar
                'text_color': "#F0F0F0",                    # Crisp, clear text
                'text_color_bright': "#F3F3F3",             # Extra bright for emphasis
                'camera_bg': "#131720",                     # Unified panel color
                'menu_bg': "#131720",                       # Consistent panels
                'scroll_bg': "#131720",                     # Unified background
                'text_edit_bg': "#1a1f2e",                  # Slightly elevated
                'border_color': "#1e293b",                  # Subtle borders
                'shadow_dark': "rgba(0, 0, 0, 0.7)",        # Deep shadows
                'shadow_light': "rgba(0, 0, 0, 0.4)",       # Soft depth
                'disabled_bg': "#1e293b",                   # Muted disabled
                'disabled_text': "#64748b",                 # Clear disabled state
                'accent_purple': "#8b5cf6",                 # Purple accent option
                'accent_cyan': "#06b6d4"                    # Cyan accent option
            }
        else:
            return {
                # Light mode: Clean, modern palette with warm undertones
                'bg_color': "#FFFFFF",                      # Softer white
                'top_bar_color': "#146D75",                 # Pure white
                'top_bar_gradient': "#146D75",              # Light blue tint
                'btn_primary': "#79BDBD",                   # Rich blue
                'btn_hover': "#74A5A5",                     # Deeper blue hover
                'btn_glow': "rgba(203, 243, 187, 0.5)",       # Subtle glow
                'transcription_bg': "#ffffff",              # Clean white
                'text_color': "#111925",                    # Rich, dark text
                'text_color_bright': "#FFFFFF",             # Maximum contrast
                'camera_bg': "#ffffff",                     # Pure white
                'menu_bg': "#ffffff",                       # Consistent panels
                'scroll_bg': "#ffffff",                     # Clean background
                'text_edit_bg': "#f8fafc",                  # Subtle gray
                'border_color': "#e2e8f0",                  # Soft borders
                'shadow_dark': "rgba(0, 0, 0, 0.1)",        # Light shadow
                'shadow_light': "rgba(0, 0, 0, 0.05)",      # Barely there
                'disabled_bg': "#cbd5e1",                   # Clear disabled
                'disabled_text': "#64748b",                 # Muted text
                'accent_purple': "#7c3aed",                 # Purple accent
                'accent_cyan': "#0891b2"                    # Cyan accent
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
                padding: 8px 12px;
                font-size: 13px;
                min-height: 20px;
                font-family: 'Inter', 'Segoe UI', -apple-system, system-ui, sans-serif;
            }}
            
            QComboBox:hover {{
                border-color: {colors['btn_primary']};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {colors['text_color']};
                width: 0;
                height: 0;
                margin-right: 5px;
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