from PySide6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QPushButton, QLabel, QMessageBox
)
from custom_popups import show_error, show_warning, show_info, show_success, StyledDialog
from styling import ThemeManager
from PySide6.QtCore import Qt
from pathlib import Path
import json
import bcrypt

USER_PREF_FILE = Path("user_preferences.json")

if not USER_PREF_FILE.exists():
    USER_PREF_FILE.write_text(json.dumps({}))

def load_user_data():
    with open(USER_PREF_FILE, "r") as f:
        return json.load(f)

def save_user_data(data):
    with open(USER_PREF_FILE, "w") as f:
        json.dump(data, f, indent=4)


class LoginDialog(QDialog):
    """Login dialog for existing users"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login - Echo Me")
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setFixedSize(320, 200)
        
        self.logged_in_user = None
        self.user_data = load_user_data()
        
        layout = QFormLayout(self)
        
        # Title
        title = QLabel("Login to Echo Me")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addRow(title)
        
        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        layout.addRow(QLabel("Username:"), self.username_input)
        
        # Password field
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.returnPressed.connect(self.attempt_login)
        layout.addRow(QLabel("Password:"), self.password_input)
        
        # Login button
        self.login_btn = QPushButton("Login")
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #008080;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00b3b3;
            }
        """)
        self.login_btn.clicked.connect(self.attempt_login)
        layout.addRow(self.login_btn)
        
        # Switch to signup
        self.signup_link = QPushButton("Don't have an account? Sign up")
        self.signup_link.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #008080;
                border: none;
                text-decoration: underline;
                padding: 5px;
            }
            QPushButton:hover {
                color: #00b3b3;
            }
        """)
        self.signup_link.clicked.connect(self.switch_to_signup)
        layout.addRow(self.signup_link)
        
        self.apply_dialog_style()
    
    def apply_dialog_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #008080;
            }
        """)
    
    def attempt_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().encode("utf-8")
        
        if not username or not password:
            show_warning(self, "Error", "Username and password cannot be empty", True)
            return
        
        if username not in self.user_data:
            show_warning(self, "Error", "Username not found. Please sign up first.", True)
            return
        
        hashed_pw = self.user_data[username]["password"].encode("utf-8")
        
        if bcrypt.checkpw(password, hashed_pw):
            self.logged_in_user = username
            self.accept()
        else:
            show_error(self, "Error", "Incorrect password", True)
            self.password_input.clear()
            self.password_input.setFocus()
    
    def switch_to_signup(self):
        """Close this dialog and open signup dialog"""
        self.hide()
        signup = SignupDialog(self.parent())
        if signup.exec() == QDialog.Accepted:
            self.logged_in_user = signup.logged_in_user
            self.accept()
        else:
            self.show()


class SignupDialog(QDialog):
    """Signup dialog for new users"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign Up - Echo Me")
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setFixedSize(320, 240)
        
        self.logged_in_user = None
        self.user_data = load_user_data()
        
        layout = QFormLayout(self)
        
        # Title
        title = QLabel("Create New Account")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addRow(title)
        
        # Username field
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Choose a username")
        layout.addRow(QLabel("Username:"), self.username_input)
        
        # Password field
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Choose a password")
        layout.addRow(QLabel("Password:"), self.password_input)
        
        # Confirm password field
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.confirm_password_input.setPlaceholderText("Confirm your password")
        self.confirm_password_input.returnPressed.connect(self.attempt_signup)
        layout.addRow(QLabel("Confirm:"), self.confirm_password_input)
        
        # Signup button
        self.signup_btn = QPushButton("Sign Up")
        self.signup_btn.setStyleSheet("""
            QPushButton {
                background-color: #008080;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00b3b3;
            }
        """)
        self.signup_btn.clicked.connect(self.attempt_signup)
        layout.addRow(self.signup_btn)
        
        # Switch to login
        self.login_link = QPushButton("Already have an account? Login")
        self.login_link.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #008080;
                border: none;
                text-decoration: underline;
                padding: 5px;
            }
            QPushButton:hover {
                color: #00b3b3;
            }
        """)
        self.login_link.clicked.connect(self.switch_to_login)
        layout.addRow(self.login_link)
        
        self.apply_dialog_style()
    
    def apply_dialog_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #333;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #008080;
            }
        """)
    
    def attempt_signup(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        
        # Validation
        if not username or not password:
            show_warning(self, "Error", "Username and password cannot be empty", True)
            return
        
        if len(username) < 3:
            show_warning(self, "Error", "Username must be at least 3 characters long", True)
            return
        
        if len(password) < 6:
            show_warning(self, "Error", "Password must be at least 6 characters long", True)
            return
        
        if password != confirm_password:
            show_warning(self, "Error", "Passwords do not match", True)
            self.password_input.clear()
            self.confirm_password_input.clear()
            self.password_input.setFocus()
            return
        
        if username in self.user_data:
            show_warning(self, "Error", "Username already exists. Please choose another.", True)
            self.username_input.clear()
            self.username_input.setFocus()
            return
        
        # Create new user
        hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        self.user_data[username] = {
            "password": hashed_pw.decode("utf-8"),
            "preferences": {
                "dark_mode": True,
                "show_landmarks": True,
                "tts_translation": "No Translation",
                "tts_voice": "English (US)",
                "tts_speed": "Normal"
            }
        }
        save_user_data(self.user_data)
        
        self.logged_in_user = username
        show_success(self, "Success", f"Account created successfully!\nWelcome, {username}!", True, True)
        self.accept()
    
    def switch_to_login(self):
        """Close this dialog and open login dialog"""
        self.hide()
        login = LoginDialog(self.parent())
        if login.exec() == QDialog.Accepted:
            self.logged_in_user = login.logged_in_user
            self.accept()
        else:
            self.show()


def show_login_flow(parent=None):
    """
    Main entry point for login/signup flow.
    Returns username if successful, None if cancelled.
    """
    login = LoginDialog(parent)
    if login.exec() == QDialog.Accepted:
        return login.logged_in_user
    return None