import argparse
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_image

TFLITE_MODEL_PATH = 'models/dr_detection_model.tflite'
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def predict(image_path):
    """Loads TFLite model and predicts on a single image."""
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = preprocess_image(image_path)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = output_data[0][predicted_class_index]

    print("\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_class_name} (Class {predicted_class_index})")
    print(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Diabetic Retinopathy from an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image file.')
    args = parser.parse_args()
    predict(args.image_path)
