"""Lightweight DCGAN-style comparison baseline (not the main VAE editor)."""
import tensorflow as tf


def build_lightweight_generator(latent_dim: int = 16):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 32, activation="relu")(inputs)
    x = tf.keras.layers.Reshape((7, 7, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="lightweight_gan_generator")
