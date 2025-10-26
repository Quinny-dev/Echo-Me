"""
User Data Management Module
Handles loading and saving user preferences and data
"""

import json
from pathlib import Path

USER_PREF_FILE = Path("user_preferences.json")
READY_FILE = Path("gui_ready.flag")


def load_user_data():
    """Load user data from JSON file"""
    try:
        if not USER_PREF_FILE.exists():
            return {}
        with open(USER_PREF_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_user_data(data):
    """Save user data to JSON file"""
    try:
        with open(USER_PREF_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving user data: {e}")
        return False


def get_user_preferences(username):
    """Get preferences for a specific user"""
    data = load_user_data()
    return data.get(username, {}).get("preferences", {})


def save_user_preferences(username, preferences):
    """Save preferences for a specific user"""
    data = load_user_data()
    if username not in data:
        data[username] = {"preferences": {}}
    elif "preferences" not in data[username]:
        data[username]["preferences"] = {}
    
    data[username]["preferences"].update(preferences)
    return save_user_data(data)


def signal_ready():
    """Signal that the GUI is ready"""
    try:
        READY_FILE.write_text("ready")
        return True
    except Exception:
        return False


def cleanup_ready_signal():
    """Clean up the ready signal file"""
    try:
        if READY_FILE.exists():
            READY_FILE.unlink()
        return True
    except Exception:
        return False