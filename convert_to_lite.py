import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("model_stone_paper_scissors.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model_stone_paper_scissors.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model_stone_paper_scissors.tflite")
