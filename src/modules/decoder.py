import tensorflow as tf

from src.modules.attention import CrossAttention


class Decoder(tf.keras.layers.Layer):

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(), mask_token='',
                                                       oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(vocabulary=text_processor.get_vocabulary(), mask_token='',
                                                       oov_token='[UNK]',
                                                       invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = CrossAttention(units)
        self.last_attention_weights = None
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, **kwargs):
        context, x = inputs
        state = kwargs.get('state', None)
        return_state = kwargs.get('return_state', False)

        x = self.embedding(x)
        x, state = self.rnn(x, initial_state=state)
        x = self.attention((x, context))
        self.last_attention_weights = self.attention.last_attention_weights
        results = self.output_layer(x)

        if return_state:
            return results, state
        else:
            return results
