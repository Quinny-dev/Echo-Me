# Virtual Environment Setup — Echo-Me

This guide walks you through setting up a Python virtual environment for the `Echo-Me` project.  
It supports both **Windows** and **macOS/Linux** and requires minimal steps to get started.

---

## Requirements

Before you begin, make sure you have:

- `Python 3.8 or higher` (Recommended: 3.10–3.12)
- `pip` (Python package manager)
- Optional: **Git** (for version control)
- Optional: **CUDA 11.8+** (for GPU training)

---

## File Overview

Inside the `setup/` folder, you’ll find:

- `setup_venv.bat` – for **Windows** users  
- `setup_venv.sh` – for **macOS/Linux** users  

And in the **project root**, you’ll find:

- `requirements.txt` – list of Python dependencies  

---

## Setup on Windows

1. Navigate to the `setup` directory.
2. Double-click the `setup_venv.bat` file or run it manually from the terminal:
   ```cmd
   setup\setup_venv.bat
   ```

The script will automatically:

- Create a virtual environment in a folder named `Echo-Me`.
- Upgrade pip.
- Install all dependencies from `requirements.txt`.
- Verify the core libraries (TensorFlow & MediaPipe).

Once it finishes, your environment is ready to use!

---

## Setup on macOS / Linux

1. Open a terminal in the project root directory.
2. Double-click the `setup_venv.sh` file or run it manually from the terminal:
   ```bash
   bash setup/setup_venv.sh
   ```

If you get a "permission denied" error, make it executable first:

```bash
chmod +x setup/setup_venv.sh
./setup/setup_venv.sh
```

The script will:

- Create the Echo-Me virtual environment.
- Activate it.
- Upgrade pip.
- Install all dependencies from `requirements.txt`.
- Confirm that TensorFlow and MediaPipe were installed successfully.

---

## Activating the Environment

After setup, you can activate the environment anytime by running this in the root folder:

**Windows:**

```cmd
setup\venv\Scripts\activate
```

**macOS / Linux:**

```bash
source setup/venv/bin/activate
```

---

## Tip for VS Code Users

To use this virtual environment inside VS Code:

1. Open the `Echo-Me` project folder.
2. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
3. Run: **Python: Select Interpreter**
4. Choose the interpreter located inside the Echo-Me folder.

VS Code will now automatically use your project's virtual environment for debugging and execution.

---

## Troubleshooting

If installation fails or dependencies conflict:

- Make sure you're using the correct Python version (`python --version`).
- Try removing the old environment and re-running setup:

```bash
rmdir /s /q Echo-Me  # Windows
rm -rf Echo-Me       # macOS/Linux
```

- Check GPU compatibility for TensorFlow:
  https://www.tensorflow.org/install/pip

---

## Next Steps

Your environment is now ready! Continue with:

- **Data Preparation** - Process and clean your dataset.
- **Model Training** - Train the LSTM model using MediaPipe landmarks.
- **Live Testing** - Run the real-time Sign-to-Speech translator.

For detailed instructions, see the main `README.md`.

---

Happy signing and coding!
— Echo-Me Development Team