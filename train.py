import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# --- CORRECTED IMPORT ---
# This now imports the correct function name from your updated preprocessing script
from src.preprocessing import prepare_image_for_aug
from src.augmentation import get_mild_augmentations, get_aggressive_augmentations
from src.build_model import build_dr_model

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
DATA_DIR = 'data'
IMAGE_FOLDER = os.path.join(DATA_DIR, 'aptos2019', 'train_images')

class SpecializedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, image_dir, batch_size, img_size, mild_augs, aggressive_augs, shuffle=True):
        self.df = df
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.mild_augs = mild_augs
        self.aggressive_augs = aggressive_augs
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        df_batch = self.df.iloc[indexes]
        
        X, y = self.__data_generation(df_batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_batch):
        X = np.empty((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        for i, (idx, row) in enumerate(df_batch.iterrows()):
            img_path = os.path.join(self.image_dir, row['id_code'])
            # --- USES THE CORRECT FUNCTION ---
            image = prepare_image_for_aug(img_path) 
            label = row['diagnosis']
            
            if image is None:
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) 
                label = 0
            
            if label == 0:
                augmented = self.mild_augs(image=image)
                image = augmented['image']
            else:
                augmented = self.aggressive_augs(image=image)
                image = augmented['image']
            
            final_image = image.astype('float32') / 255.0

            X[i,] = final_image
            y[i] = label
            
        return X, y

# --- Main Training Logic ---
def train():
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_split.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_split.csv'))
    except FileNotFoundError:
        print("!!! ERROR: train_split.csv or val_split.csv not found. Please run 'python src/prepare_data.py' first. !!!")
        return

    mild_augs = get_mild_augmentations()
    aggressive_augs = get_aggressive_augmentations()

    train_gen = SpecializedDataGenerator(train_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE, 
                                       mild_augs=mild_augs, aggressive_augs=aggressive_augs)
    
    val_gen = SpecializedDataGenerator(val_df, IMAGE_FOLDER, BATCH_SIZE, IMG_SIZE,
                                     mild_augs=mild_augs, aggressive_augs=mild_augs, shuffle=False)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
    class_weights = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {class_weights}")

    model = build_dr_model()

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/best_model_v2.h5', 
        save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        mode='max', restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.2, patience=7,
        min_lr=1e-6, mode='max', verbose=1
    )

    print("\n--- Starting model training with V2 - Targeted Augmentation... ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )
    
    model.save('models/final_model_v2.h5')
    print("--- Final model V2 successfully saved to models/final_model_v2.h5 ---")
    
    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.title("Training History (V2 - Targeted Augmentation)")
    plt.savefig('reports/figures/training_history_v2.png')
    print("--- Training history plot V2 saved. ---")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train()

