import tensorflow as tf


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layer_normalization = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        self.last_attention_weights = None

    def call(self, inputs, **kwargs):
        x, context = inputs
        mask = kwargs.get('mask', None)
        attn_output, attn_scores = self.mha(query=x, value=context, key=context, attention_mask=mask,
                                            return_attention_scores=True)
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores
        x = self.add([x, attn_output])
        x = self.layer_normalization(x)

        return x
