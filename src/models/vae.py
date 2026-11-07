import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon


@tf.keras.utils.register_keras_serializable(package="latentstyle")
class VAE(tf.keras.Model):
    def __init__(self, latent_dim=16, image_shape=(28, 28, 1), beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.beta = beta

        inputs = tf.keras.Input(shape=image_shape)
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        mean = tf.keras.layers.Dense(latent_dim, name="latent_mean")(x)
        log_var = tf.keras.layers.Dense(latent_dim, name="latent_log_var")(x)
        z = Sampling(name="z")([mean, log_var])
        self.encoder = tf.keras.Model(inputs, [mean, log_var, z], name="encoder")

        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((7, 7, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
        self.decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

    def call(self, inputs, training=None):
        mean, log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def encode_mean(self, inputs):
        return self.encoder(inputs, training=False)[0]

    def encode(self, inputs):
        return self.encoder(inputs, training=False)[2]

    def reconstruct(self, inputs):
        mean, _, _ = self.encoder(inputs, training=False)
        return self.decoder(mean, training=False)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mean, log_var, _ = self.encoder(x, training=True)
            reconstruction = self.decoder(mean + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mean)), training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
            )
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        mean, log_var, _ = self.encoder(x, training=False)
        reconstruction = self.decoder(mean, training=False)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
        )
        return {"loss": reconstruction_loss + self.beta * kl_loss,
                "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}
