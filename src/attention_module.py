import tensorflow as tf
from tensorflow.keras import layers

def channel_attention_module(x, ratio=8):
    """Channel Attention Module (CAM)."""
    ch = x.shape[-1]
    shared_mlp = tf.keras.Sequential([
        layers.Dense(ch // ratio, activation='relu'),
        layers.Dense(ch)
    ])

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, ch))(avg_pool)
    avg_out = shared_mlp(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, ch))(max_pool)
    max_out = shared_mlp(max_pool)

    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation('sigmoid')(attention)
    return layers.Multiply()([x, attention])

def spatial_attention_module(x):
    """Spatial Attention Module (SAM)."""
    avg_pool = tf.reduce_mean(x, axis=3, keepdims=True)
    max_pool = tf.reduce_max(x, axis=3, keepdims=True)
    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
    attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid',
                              kernel_initializer='he_normal', use_bias=False)(concat)
    return layers.Multiply()([x, attention])

def cbam_block(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    return x
