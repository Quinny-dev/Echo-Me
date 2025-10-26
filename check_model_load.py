# Run this once to convert your model:
import tensorflow as tf

model = tf.keras.layers.TFSMLayer('models/model_fast', call_endpoint='serving_default')
wrapped = tf.keras.Sequential([model])
wrapped.save('models/model_fast.keras')  # Save in new format