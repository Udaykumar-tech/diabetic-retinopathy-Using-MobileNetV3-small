import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from src.preprocessing import preprocess_image

MODEL_PATH = 'models/best_model.h5'
LAST_CONV_LAYER_NAME = 'conv_1'  # From MobileNetV3Small summary
IMG_SIZE = 224

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(image_path, heatmap, output_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(output_path)

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    test_df = pd.read_csv('data/test_split.csv')
    sample_df = test_df.sample(5)
    output_dir = 'reports/grad_cam'
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Grad-CAM visualizations...")
    for _, row in sample_df.iterrows():
        img_name = row['id_code']
        img_path = os.path.join('data/aptos2019/train_images', img_name)
        img_array = preprocess_image(img_path)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        heatmap = make_gradcam_heatmap(img_array_expanded, model, LAST_CONV_LAYER_NAME)
        output_path = os.path.join(output_dir, f'gradcam_{img_name}')
        save_gradcam(img_path, heatmap, output_path)
    print(f"Grad-CAM images saved in {output_dir}")

if __name__ == '__main__':
    main()
