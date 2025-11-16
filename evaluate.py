import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# We now need to import the new generator and one augmentation function
from train import SpecializedDataGenerator, get_mild_augmentations

# --- Configuration ---
# CHANGE THIS VARIABLE TO TEST YOUR DIFFERENT MODELS
MODEL_TO_EVALUATE = 'models/final_model_v2.h5' 
# To test your first model, change the line above to: 'models/final_model.h5'

DATA_DIR = 'data'
REPORTS_DIR = 'reports'
BATCH_SIZE = 32
IMG_SIZE = 224

def evaluate_model(model, df, image_dir, dataset_name, model_name_tag):
    """
    Evaluates the model on a given dataframe and saves the results
    with a name specific to the model version being tested.
    """
    print(f"\n--- Evaluating {model_name_tag} on {dataset_name} ---")
    
    # Create a data generator for the test set (no aggressive augmentations)
    mild_augs = get_mild_augmentations()
    test_gen = SpecializedDataGenerator(df, image_dir, BATCH_SIZE, IMG_SIZE, 
                                      mild_augs=mild_augs, aggressive_augs=mild_augs, shuffle=False)
    
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = df['diagnosis'].values[:len(y_pred)]
    
    report = classification_report(y_true, y_pred)
    print(f"Classification Report for {dataset_name}:\n{report}")
    # Save report with a model-specific name to avoid overwriting
    report_path = os.path.join(REPORTS_DIR, f'classification_report_{dataset_name}_{model_name_tag}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Results for model: {model_name_tag}\n\n")
        f.write(report)
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.title(f'Confusion Matrix - {dataset_name} ({model_name_tag})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # Save confusion matrix with a model-specific name
    cm_path = os.path.join(REPORTS_DIR, 'figures', f'confusion_matrix_{dataset_name}_{model_name_tag}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()

def main():
    print(f"--- Loading model for evaluation: {MODEL_TO_EVALUATE} ---")
    # It's good practice to re-compile the model after loading, although not always necessary for inference
    model = tf.keras.models.load_model(MODEL_TO_EVALUATE, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create a simple tag from the model filename for saving reports
    model_name_tag = os.path.basename(MODEL_TO_EVALUATE).replace('.h5', '')

    # 1. Evaluate on APTOS 2019 test set
    test_df_aptos = pd.read_csv(os.path.join(DATA_DIR, 'test_split.csv'))
    image_dir_aptos = os.path.join(DATA_DIR, 'aptos2019', 'train_images')
    evaluate_model(model, test_df_aptos, image_dir_aptos, 'APTOS_2019', model_name_tag)
    
    # 2. Cross-dataset validation on EyePACS
    eyepacs_csv_path = os.path.join(DATA_DIR, 'eyepacs', 'trainLabels.csv')
    if os.path.exists(eyepacs_csv_path):
        test_df_eyepacs = pd.read_csv(eyepacs_csv_path)
        test_df_eyepacs.columns = ['id_code', 'diagnosis']
        test_df_eyepacs['id_code'] = test_df_eyepacs['id_code'] + '.png' # Assuming PNG format
        image_dir_eyepacs = os.path.join(DATA_DIR, 'eyepacs', 'train_images')
        # Using a subset for faster evaluation, you can use the full set if desired
        evaluate_model(model, test_df_eyepacs.head(1000), image_dir_eyepacs, 'EyePACS', model_name_tag)
    else:
        print("\nEyePACS dataset not found. Skipping cross-dataset validation.")

if __name__ == '__main__':
    main()

