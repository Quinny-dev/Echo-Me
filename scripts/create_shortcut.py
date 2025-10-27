import os
import platform
from pathlib import Path

READY_FILE = "gui_ready.flag"

def create_launcher():
    base_dir = Path(__file__).resolve().parents[1]
    launcher_path = base_dir / "scripts" / "launcher.py"
    logo_path = base_dir / "assets" / "Echo_Me_Logo.ico"

    # Prefer JPEG background if present, otherwise fall back to JPG/PNG at runtime
    launcher_code = """
import subprocess
import tkinter as tk
from PIL import Image, ImageTk
import threading
import math
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
READY_FILE = BASE_DIR / "gui_ready.flag"

def splash_screen():
    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry("400x300")

    canvas = tk.Canvas(root, width=400, height=300, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Try loading a background image from assets (jpeg/jpg/png)
    bg_photo = None
    for name in ("Echo_Me_Logo.jpeg", "Echo_Me_Logo.jpg", "Echo_Me_Logo.png"):
        candidate = BASE_DIR / "assets" / name
        if candidate.exists():
            try:
                img = Image.open(candidate)
                img = img.resize((400, 300))
                bg_photo = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, image=bg_photo, anchor="nw")
                canvas.image = bg_photo
                break
            except Exception:
                pass

    if bg_photo is None:
        root.configure(bg="white")
        canvas.configure(bg="white")
        canvas.create_text(200, 120, text="Loading Echo Me...", font=("Arial", 14), fill="#008080")

    # Rotating loader animation
    radius = 5
    offset_x, offset_y = 200, 262
    angles = [0, 90, 180, 270]
    circles = []
    for angle in angles:
        x = offset_x + math.cos(math.radians(angle)) * 15
        y = offset_y + math.sin(math.radians(angle)) * 15
        circle = canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="#FFFFFF", outline="")
        circles.append(circle)

    def rotate():
        for i, angle in enumerate(angles):
            angles[i] += 10
            x = offset_x + math.cos(math.radians(angles[i])) * 15
            y = offset_y + math.sin(math.radians(angles[i])) * 15
            canvas.coords(circles[i], x-radius, y-radius, x+radius, y+radius)
        root.after(50, rotate)

    rotate()

    # Center window
    root.eval('tk::PlaceWindow . center')
    
    # Delete old ready file if it exists
    if READY_FILE.exists():
        try:
            READY_FILE.unlink()
        except Exception:
            pass

    # Launch GUI in background
    pythonw_path = BASE_DIR / "setup" / "venv" / "Scripts" / "pythonw.exe"
    gui_path = BASE_DIR / "scripts" / "gui.py"
    
    subprocess.Popen([str(pythonw_path), str(gui_path)], shell=False, cwd=str(BASE_DIR))

    # Wait for GUI ready flag, then close splash
    def wait_for_gui_ready():
        max_wait = 60
        wait_time = 0
        check_interval = 0.2
        
        while not READY_FILE.exists() and wait_time < max_wait:
            time.sleep(check_interval)
            wait_time += check_interval
        
        if READY_FILE.exists():
            time.sleep(0.5)
        
        root.destroy()

    threading.Thread(target=wait_for_gui_ready, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    splash_screen()
"""

    launcher_path.write_text(launcher_code, encoding="utf-8")
    print(f"Created launcher.py at {launcher_path}")


def create_shortcut():
    system = platform.system()
    base_dir = Path(__file__).resolve().parents[1]
    launcher_path = base_dir / "scripts" / "launcher.py"
    venv_pythonw = base_dir / "setup" / "venv" / "Scripts" / "pythonw.exe"
    logo_path = base_dir / "assets" / "Echo_Me_Logo.ico"

    create_launcher()

    if system == "Windows":
        import winshell
        from win32com.client import Dispatch

        desktop = Path(winshell.desktop())
        shortcut_path = desktop / "Echo-Me.lnk"

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(shortcut_path))
        shortcut.TargetPath = str(venv_pythonw)
        shortcut.Arguments = f'"{launcher_path}"'
        shortcut.WorkingDirectory = str(base_dir)
        shortcut.IconLocation = str(logo_path.resolve())
        shortcut.save()

        print(f"Shortcut created on Desktop: {shortcut_path}")

    # macOS shortcut creation
    elif system == "Darwin":
        shortcut_path = Path.home() / "Desktop" / "Echo-Me.command"
        with open(shortcut_path, "w") as f:
            f.write(f"#!/bin/bash\\nsource {base_dir}/setup/venv/bin/activate\\npython3 '{launcher_path}'\\n")
        # Make executable
        os.chmod(shortcut_path, 0o755)
        print(f"macOS launcher created: {shortcut_path}")

    # Linux desktop file creation
    else:
        shortcut_path = Path.home() / "Desktop" / "Echo-Me.desktop"
        with open(shortcut_path, "w") as f:
            f.write(f"""[Desktop Entry]
Type=Application
Terminal=false
Exec=bash -c 'source "{base_dir}/setup/venv/bin/activate" && python3 "{launcher_path}"'
Name=Echo Me
Icon={logo_path}
""")
        # Make executable
        os.chmod(shortcut_path, 0o755)
        print(f"Linux desktop shortcut created: {shortcut_path}")


# Run shortcut creation when script is executed directly
if __name__ == "__main__":
    create_shortcut()