import numpy as np
import pytest
import tensorflow as tf

from src.modules.embedding import PositionalEmbedding

tf.random.set_seed(42)
np.random.seed(42)

INPUT_DIM = 10000
OUTPUT_DIM = 512
BATCH_SIZE = 2
SEQUENCE_LENGTH = 2048


@pytest.fixture(scope="module")
def setup_positional_embedding():
    model = PositionalEmbedding(INPUT_DIM, OUTPUT_DIM)
    input_sequence = tf.random.uniform((BATCH_SIZE, SEQUENCE_LENGTH), minval=0, maxval=INPUT_DIM, dtype=tf.int32)
    output_sequence = model(input_sequence)
    return model, input_sequence, output_sequence


def test_positional_embedding_shape(setup_positional_embedding):
    model, input_sequence, output_sequence = setup_positional_embedding
    assert output_sequence.shape == (BATCH_SIZE, SEQUENCE_LENGTH, OUTPUT_DIM)


def test_positional_embedding_values(setup_positional_embedding):
    model, input_sequence, output_sequence = setup_positional_embedding

    embedded_sequence = model.embedding(input_sequence)
    embedded_sequence *= tf.math.sqrt(tf.cast(model.output_dim, tf.float32))

    length = tf.shape(input_sequence)[1]
    pos_encoding = model.pos_encoding[tf.newaxis, :length, :]
    expected_output = embedded_sequence + pos_encoding

    np.testing.assert_almost_equal(output_sequence.numpy(), expected_output.numpy(), decimal=5)
