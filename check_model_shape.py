#-- I used this to check what labels are being loaded 
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

model_dir = "models/model_fast"
model = tf.keras.models.load_model(model_dir)

print("\n=== Model Summary ===")
model.summary()

print("\n=== Model Config ===")
print(model.get_config())

# Print all layers to inspect attributes
for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}: {layer.name}")
    try:
        print("  Config:", layer.get_config())
    except:
        print("  No config available")

# Check custom objects or attributes
if hasattr(model, 'classes_'):
    print("\nClasses attribute found:", model.classes_)