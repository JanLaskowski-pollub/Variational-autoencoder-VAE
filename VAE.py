import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers 
import matplotlib.pyplot as plt
import sklearn

df = pd.read_excel('../OV/Badanie_pokolenia_silver (55+).xlsx')
df = df.drop(['LP','M4.txt. Inne, jakie?'], axis=1)
df = df.replace('^(.*)\.\s*.*', r'\1', regex=True)
df['M3. Miejsce zamieszkania'] = df['M3. Miejsce zamieszkania'].apply(lambda x: x.split('.')[0])
metryczka = ['M1. Płeć','M2. Wiek','M3. Miejsce zamieszkania','M4. Miejsce zatrudnienia (branża)','M5. Wykształcenie']
BlokI = df.columns[5:21]
metr = df[metryczka]
dfX = df[BlokI]
std = dfX.std(axis=1)

dfX = pd.concat([dfX, std], axis=1)

dfX = dfX[dfX[0] > 1]
dfX = dfX.drop([0], axis=1)

dfX = dfX.reset_index(drop=True)

X = np.array(dfX.values)
X = np.expand_dims(X,axis=-2)


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
encoder_inputs = tf.keras.Input(shape=(1, 16))

x = tf.keras.layers.Conv1D(12, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
x = tf.keras.layers.Conv1D(8, 4, activation="relu", strides=2, padding="same")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4, activation="relu")(x)
z_mean = tf.keras.layers.Dense(2, name="z_mean")(x)
z_log_var = tf.keras.layers.Dense(2, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Reshape((1,2))(latent_inputs)
x = tf.keras.layers.Conv1DTranspose(4, 4, activation="relu", strides=2, padding="same")(x)
x = tf.keras.layers.Conv1DTranspose(8, 4, activation="relu", strides=2, padding="same")(x)
x = tf.keras.layers.Conv1DTranspose(12, 4, activation="relu", strides=2, padding="same")(x)
x = tf.keras.layers.Flatten()(x)
decoder_outputs = tf.keras.layers.Dense(16, activation="relu")(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

X = X.astype(np.float32)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker =  tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker =  tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.losses.MeanSquaredError()(data, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(X, epochs=400, batch_size=128)