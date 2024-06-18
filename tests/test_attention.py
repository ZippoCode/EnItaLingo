import pytest
import tensorflow as tf

from src.modules.attention import CrossAttention

UNITS = 64
BATCH_SIZE = 32
SEQUENCE_LENGTH_SOURCE = 10
SEQUENCE_LENGTH_TARGET = 15


@pytest.fixture(scope="module")
def setup_attention_layer():
    target_text_processor = tf.keras.layers.TextVectorization(max_tokens=100,
                                                              output_sequence_length=SEQUENCE_LENGTH_TARGET,
                                                              ragged=True)
    example_texts = ["This is a test.", "Another text example."]
    target_text_processor.adapt(example_texts)
    embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(), output_dim=UNITS, mask_zero=True)
    attention_layer = CrossAttention(units=UNITS)
    ex_context = tf.random.uniform(shape=[BATCH_SIZE, SEQUENCE_LENGTH_SOURCE, UNITS])
    ex_tar_in = tf.random.uniform(shape=[BATCH_SIZE, SEQUENCE_LENGTH_TARGET], minval=0,
                                  maxval=target_text_processor.vocabulary_size(), dtype=tf.int32)

    return embed, attention_layer, ex_context, ex_tar_in


def non_test_attention_layer_output_shape(setup_attention_layer):  # TO IMPROVE
    embed, attention_layer, ex_context, ex_tar_in = setup_attention_layer
    ex_tar_embed = embed(ex_tar_in)
    result = attention_layer((ex_tar_embed, ex_context))

    assert ex_context.shape == (BATCH_SIZE, SEQUENCE_LENGTH_SOURCE, UNITS)
    assert ex_tar_embed.shape == (BATCH_SIZE, SEQUENCE_LENGTH_TARGET, UNITS)
    assert result.shape == (BATCH_SIZE, SEQUENCE_LENGTH_TARGET, UNITS)
    assert attention_layer.last_attention_weights.shape == (BATCH_SIZE, SEQUENCE_LENGTH_TARGET, SEQUENCE_LENGTH_SOURCE)


def test_cross_attention_initialization(setup_attention_layer):
    _, attention_layer, _, _ = setup_attention_layer

    assert isinstance(attention_layer.mha, tf.keras.layers.MultiHeadAttention)
    assert isinstance(attention_layer.layer_normalization, tf.keras.layers.LayerNormalization)
    assert isinstance(attention_layer.add, tf.keras.layers.Add)


def test_cross_attention_call(setup_attention_layer):
    _, attention_layer, _, _ = setup_attention_layer

    x = tf.random.uniform(shape=[1, SEQUENCE_LENGTH_SOURCE, UNITS])
    context = tf.random.uniform(shape=[1, SEQUENCE_LENGTH_SOURCE, UNITS])
    output = attention_layer((x, context))
    assert output.shape == (1, SEQUENCE_LENGTH_SOURCE, UNITS)
    assert attention_layer.last_attention_weights is not None
