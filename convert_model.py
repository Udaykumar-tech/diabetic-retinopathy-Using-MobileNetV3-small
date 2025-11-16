import tensorflow as tf
import os

MODEL_PATH = 'models/best_model.h5'
OUTPUT_TFLITE_PATH = 'models/dr_detection_model.tflite'

def convert_to_tflite():
    print(f"Loading Keras model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(OUTPUT_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Model successfully converted to TFLite format and saved at: {OUTPUT_TFLITE_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_TFLITE_PATH) / 1024:.2f} KB")

if __name__ == '__main__':
    convert_to_tflite()
