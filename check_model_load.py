# Run this once to convert your model:
import tensorflow as tf

# Load the SavedModel format as a TFSMLayer
model = tf.keras.layers.TFSMLayer('models/model_fast', call_endpoint='serving_default')
# Wrap it in a Sequential model
wrapped = tf.keras.Sequential([model])
# Save in new Keras format for easier loading
wrapped.save('models/model_fast.keras')