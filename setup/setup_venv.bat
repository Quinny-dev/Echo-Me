@echo off
chcp 65001 >nul
echo ============================================================
echo Echo-Me Environment Setup
echo ============================================================
echo.

:: Navigate to the project root (one level up from setup/)
cd /d "%~dp0.."

echo [INFO] Creating virtual environment with portable Python...
setup\python-portable\python.exe -m venv setup\venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call setup\venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed, continuing anyway.
)

:: Install packages that fail to build from source first
echo [INFO] Installing PyAudio and bcrypt wheels to avoid compilation...
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary :all: PyAudio bcrypt
if errorlevel 1 (
    echo [WARNING] Some wheels failed, pip will try to build from source.
)

echo [INFO] Installing remaining dependencies from requirements.txt...
pip install -r "%~dp0..\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check requirements.txt.
    pause
    exit /b 1
)

echo [INFO] Verifying TensorFlow installation...
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__')" || (
    echo [WARNING] TensorFlow not found!
)

echo [INFO] Verifying MediaPipe installation...
python -c "import mediapipe as mp; print('MediaPipe Version:', mp.__version__')" || (
    echo [WARNING] MediaPipe not found!
)

echo.
echo ============================================================
echo [SUCCESS] Setup complete!
echo To activate the environment, run:
echo.
echo     setup\venv\Scripts\activate
echo ============================================================
pause