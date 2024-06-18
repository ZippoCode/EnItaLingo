import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(input_dim, output_dim, mask_zero=True)
        self.pos_encoding = self._positional_encoding(length=2048)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.output_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def _positional_encoding(self, length):
        depth = self.output_dim / 2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth

        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)
