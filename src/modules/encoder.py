import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_vectorization, units):
        super(Encoder, self).__init__()
        self.text_processor = text_vectorization
        self.vocab_size = text_vectorization.vocabulary_size()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum',
                                                 layer=tf.keras.layers.GRU(units,
                                                                           return_sequences=True,
                                                                           recurrent_initializer='glorot_uniform'))

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context
