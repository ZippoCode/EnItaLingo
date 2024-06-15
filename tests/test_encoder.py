import numpy as np
import tensorflow as tf

from src.models.encoder import Encoder

text_processor = None
encoder = None
units = None
batch_size = None
num_ids = None


def setup_module(module):
    global text_processor, encoder, units, batch_size, num_ids
    text_processor = tf.keras.layers.TextVectorization(max_tokens=100, output_sequence_length=20, ragged=True)
    example_texts = ["This is a test.", "Another example of text."]
    text_processor.adapt(example_texts)
    units = 64
    batch_size = 50
    num_ids = 80
    encoder = Encoder(text_processor, units)


def test_embedding_values():
    example_input = ["This is a test."]
    context = encoder.convert_input(example_input)
    assert not np.any(np.isnan(context.numpy())), "Embedding values contain NaN!"


def test_batch_inputs():
    example_input = tf.random.uniform(shape=[batch_size, num_ids])
    example_result = encoder(example_input)
    expected_shape = (batch_size, num_ids, units)
    assert example_result.shape == expected_shape, f"Output shape is different from expected. " \
                                                   f"Expected shape: {expected_shape}, but got: {example_result.shape}."


def test_call_method_with_single_example():
    example_input = tf.constant([[1, 2, 3, 0, 0]])
    output = encoder(example_input)
    assert output.shape == (1, 5, units), f"Output shape for single input is different from expected. " \
                                          f"Expected shape: (1, 5, {units}), but got: {output.shape}."


def test_convert_input_with_batch():
    example_inputs = ["This is a test.", "Another example of text."]
    contexts = encoder.convert_input(example_inputs)
    assert contexts.shape[0] == len(example_inputs), f"Batch output size does not match number of inputs. " \
                                                     f"Expected {len(example_inputs)} outputs, but got {contexts.shape[0]}."


def test_embedding_layer():
    input_ids = tf.constant([[1, 2, 3], [4, 5, 0]])
    embedding_output = encoder.embedding(input_ids)

    assert not np.any(np.isnan(embedding_output.numpy())), "Embedding output contains NaN values!"

    expected_shape = (input_ids.shape[0], input_ids.shape[1], units)
    assert embedding_output.shape == expected_shape, f"Embedding output shape is incorrect. " \
                                                     f"Expected shape: {expected_shape}, but got: {embedding_output.shape}."

    example_lookup = encoder.embedding(tf.constant([[1, 2, 0], [3, 0, 0]]))
    assert example_lookup.shape == (2, 3, units), "Embedding lookup shape is incorrect for specific example."


def test_rnn_layer():
    input_data = tf.random.uniform(shape=[batch_size, num_ids, units])

    rnn_output = encoder.rnn(input_data)

    expected_shape = (batch_size, num_ids, units)
    assert rnn_output.shape == expected_shape, f"RNN output shape is incorrect. " \
                                               f"Expected shape: {expected_shape}, but got: {rnn_output.shape}."