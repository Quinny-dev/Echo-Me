"""
User Data Management Module
Handles loading and saving user preferences and data
"""

import json
from pathlib import Path

# File paths for user data and GUI signaling
USER_PREF_FILE = Path("user_preferences.json")
READY_FILE = Path("gui_ready.flag")


def load_user_data():
    """Load user data from JSON file"""
    try:
        # Return empty dict if file doesn't exist
        if not USER_PREF_FILE.exists():
            return {}
        # Load and parse JSON data
        with open(USER_PREF_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Return empty dict if file is corrupted or missing
        return {}


def save_user_data(data):
    """Save user data to JSON file"""
    try:
        # Write data to JSON file with formatting
        with open(USER_PREF_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving user data: {e}")
        return False


def get_user_preferences(username):
    """Get preferences for a specific user"""
    # Load all data and extract user's preferences
    data = load_user_data()
    return data.get(username, {}).get("preferences", {})


def save_user_preferences(username, preferences):
    """Save preferences for a specific user"""
    # Load existing data and ensure user structure exists
    data = load_user_data()
    if username not in data:
        data[username] = {"preferences": {}}
    elif "preferences" not in data[username]:
        data[username]["preferences"] = {}
    
    # Update preferences and save
    data[username]["preferences"].update(preferences)
    return save_user_data(data)


def signal_ready():
    """Signal that the GUI is ready"""
    try:
        # Create flag file to signal GUI is ready
        READY_FILE.write_text("ready")
        return True
    except Exception:
        return False


def cleanup_ready_signal():
    """Clean up the ready signal file"""
    try:
        # Remove ready flag file if it exists
        if READY_FILE.exists():
            READY_FILE.unlink()
        return True
    except Exception:
        return False