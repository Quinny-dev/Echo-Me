#!/bin/bash
echo "============================================================"
echo "Echo-Me Environment Setup"
echo "============================================================"
echo

# Navigate to project root (one level up from setup/)
cd "$(dirname "$0")/.." || exit 1

echo "[INFO] Creating virtual environment with portable Python..."
setup/python-portable/python -m venv setup/venv
if [ $? -ne 0 ]; then
  echo "[ERROR] Failed to create virtual environment."
  exit 1
fi

echo "[INFO] Activating virtual environment..."
source setup/venv/bin/activate

echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

echo "[INFO] Installing dependencies from requirements.txt..."
pip install -r "$(dirname "$0")/../requirements.txt"
if [ $? -ne 0 ]; then
  echo "[ERROR] Failed to install dependencies. Check requirements.txt"
  deactivate
  exit 1
fi

echo "[INFO] Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)" || echo "[WARNING] TensorFlow not found!"

echo "[INFO] Verifying MediaPipe installation..."
python -c "import mediapipe as mp; print('MediaPipe Version:', mp.__version__)" || echo "[WARNING] MediaPipe not found!"

echo
echo "============================================================"
echo "[SUCCESS] Setup complete!"
echo "To activate the environment, run:"
echo
echo "    source setup/venv/bin/activate"
echo "============================================================"