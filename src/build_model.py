import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, Model
from src.attention_module import cbam_block

IMG_SIZE = 224
NUM_CLASSES = 5

def build_dr_model():
    """
    Builds the CBAM-enhanced MobileNetV3 model for Diabetic Retinopathy detection.
    """
    # Base model: MobileNetV3 pre-trained on ImageNet
    base_model = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = base_model(inputs, training=True)
    
    # Apply the CBAM attention module
    x = cbam_block(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # <-- THIS IS THE CHANGE
    )
    
    print("Model built and compiled successfully.")
    model.summary()
    
    return model

if __name__ == '__main__':
    model = build_dr_model()

