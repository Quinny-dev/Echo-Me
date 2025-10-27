# Script to inspect model structure and configuration
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

# Load the model from directory
model_dir = "models/model_fast"
model = tf.keras.models.load_model(model_dir)

# Display model architecture summary
print("\n=== Model Summary ===")
model.summary()

# Display model configuration details
print("\n=== Model Config ===")
print(model.get_config())

# Iterate through all layers and display their configurations
for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    try:
        print("  Config:", layer.get_config())
    except:
        print("  No config available")

# Check if model has custom class labels attribute
if hasattr(model, 'classes_'):
    print("\nClasses attribute found:", model.classes_)