
"""
model_handler.py
Handles loading the trained sign language model and running real-time predictions.
Emits predictions via Qt signals to be displayed in the GUI.
"""
import sys
from pathlib import Path

# ✅ Dynamically add project root (Echo-Me folder) to sys.path
# scripts/model_handler.py -> scripts/ -> parent is Echo-Me
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))  # Use insert(0) for highest priority

# ✅ Now safe to import from root-level folders like `tools`
from tools.holistic import normalize_features, to_landmark_row

from PySide6.QtCore import QObject, Signal
import tensorflow as tf
import numpy as np
import traceback


DATA_PATH = Path("data")


def load_labels_from_data_folder():
    """
    Load the class labels based on folder structure in the 'data' directory.
    Follows alphabetical folder structure: None, holds_data, nonholds_data
    """
    label_dirs = []
    folder_order = ["None", "holds_data", "nonholds_data"]

    for folder_name in folder_order:
        folder_path = DATA_PATH / folder_name
        if folder_path.exists() and folder_path.is_dir():
            if folder_name == "None":
                label_dirs.append("None")
            else:
                for f in sorted(folder_path.iterdir()):
                    if f.is_dir():
                        label_dirs.append(f.name)

    print("✅ Loaded labels:", label_dirs)
    return label_dirs


class ModelHandler(QObject):
    """
    Manages the sign language recognition TensorFlow model.
    Maintains a sliding window of features and emits predictions.
    """

    # Emits: label (string), confidence (float)
    prediction_made = Signal(str, float)

    def __init__(self, model_path="models/model_fast", window_feature_size=130, frame_skip=5):
        super().__init__()
        self.model_path = model_path
        self.window_feature_size = window_feature_size
        self.frame_skip = frame_skip

        # Model properties
        self.model = None
        self.is_savedmodel = False
        self.input_key = None
        self.frame_counter = 0

        # Classification and window properties
        self.LABELS = []
        self.TIMESTEPS = 30
        self.window = None
        
        # ✅ Track last prediction to avoid duplicates
        self.last_predicted_label = None

        # Initialize model components
        self._load_model()
        self._load_labels()
        self._initialize_window()

    def _load_model(self):
        """Load the TensorFlow model with Keras-first strategy."""
        try:
            print(f"🔄 Loading model from: {self.model_path}")

            try:
                # First attempt: load as standard Keras model
                self.model = tf.keras.models.load_model(self.model_path)
                self.TIMESTEPS = self.model.input_shape[1]
                self.is_savedmodel = False
                print("✅ Model loaded using keras.models.load_model")

            except Exception as keras_error:
                print(f"⚠️ keras.models.load_model failed: {keras_error}")
                print("🔄 Attempting SavedModel format...")
                # Second attempt: load as SavedModel format
                loaded_model = tf.saved_model.load(self.model_path)
                self.model = loaded_model.signatures['serving_default']

                input_spec = list(self.model.structured_input_signature[1].values())[0]
                self.input_key = list(self.model.structured_input_signature[1].keys())[0]
                self.TIMESTEPS = int(input_spec.shape[1])
                self.is_savedmodel = True
                print("✅ Model loaded using tf.saved_model.load")

        except Exception as final_error:
            print(f"❌ Fatal: Could not load model: {final_error}")
            traceback.print_exc()
            self.model = None

    def _load_labels(self):
        """Load label names from data folder."""
        try:
            self.LABELS = load_labels_from_data_folder()
        except Exception as error:
            print(f"❌ Could not load labels: {error}")
            self.LABELS = []

    def _initialize_window(self):
        """Initialize the sliding feature window."""
        self.window = np.zeros((self.TIMESTEPS, self.window_feature_size), dtype=np.float32)
        print(f"🧠 Initialized sliding window with shape {self.window.shape}")

    def process_landmarks(self, results):
        """Process Mediapipe results object for real-time model inference with debug logging."""
        if self.model is None:
            print("🚫 Model is not loaded.")
            return

        if results is None:
            print("⚠️ No results returned from hand detector.")
            return

        # Check Mediapipe hand detection
        if not hasattr(results, "multi_hand_landmarks") or not results.multi_hand_landmarks:
            print("🖐 No hands detected in frame.")
            return

        print(f"🖐 Hands detected: {len(results.multi_hand_landmarks)}")

        try:
            # Extract features from hand landmarks
            features = to_landmark_row(results, use_holistic=False)
            print(f"📏 Raw features length: {len(features)}")

            # Normalize features for model input
            normalized = normalize_features(features)
            print(f"✅ Normalized features length: {len(normalized)}")

            # Update sliding window with new features
            self.window[:-1] = self.window[1:]
            self.window[-1] = normalized
            print(f"🧠 Updated sliding window.")

            # Skip frames to reduce processing load
            self.frame_counter += 1
            print(f"⏱ Frame counter: {self.frame_counter}/{self.frame_skip}")
            if self.frame_counter < self.frame_skip:
                return
            self.frame_counter = 0

            # Prepare input tensor for model prediction
            input_data = np.array([self.window], dtype=np.float32)
            print(f"📦 Model input shape: {input_data.shape}")

            # Run model inference
            if self.is_savedmodel:
                input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                preds = self.model(**{self.input_key: input_tensor})
                preds_array = list(preds.values())[0].numpy()[0]
            else:
                preds = self.model(input_data)
                preds_array = preds.numpy()[0] if hasattr(preds, "numpy") else preds[0]

            # Extract prediction results
            class_index = int(np.argmax(preds_array))
            confidence = float(np.max(preds_array))
            label = self.LABELS[class_index] if class_index < len(self.LABELS) else "Unknown"

            print(f"🔮 Prediction attempt: {label} ({confidence:.2f})")

            # ✅ Filter out "None", low confidence, and duplicate consecutive predictions
            if confidence > 0.5 and label != "None" and label != self.last_predicted_label:
                print(f"✅ Emitting prediction: {label} ({confidence:.2f})")
                self.prediction_made.emit(label, confidence)
                self.last_predicted_label = label  # ✅ Update last prediction
            else:
                if label == "None":
                    print(f"🚫 Filtered out 'None' prediction")
                elif label == self.last_predicted_label:
                    print(f"🔁 Filtered out duplicate prediction: {label}")
                else:
                    print(f"⚠️ Prediction confidence too low: {confidence:.2f}")

        except Exception as e:
            print(f"❌ Error during model prediction: {e}")
            import traceback
            traceback.print_exc()