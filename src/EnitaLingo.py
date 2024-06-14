import tensorflow as tf

from src.preprocessing_text import tf_normalize_text


class EnitaLingo:

    def __init__(self):
        self._source_text_vectorization = None
        self._target_text_vectorization = None

    def build_text_vectorization(self, dataset: tf.data.Dataset, max_vocabulary_size: int):
        self._source_text_vectorization = tf.keras.layers.TextVectorization(
            standardize=tf_normalize_text,
            max_tokens=max_vocabulary_size,
            ragged=True)
        self._target_text_vectorization = tf.keras.layers.TextVectorization(
            standardize=tf_normalize_text,
            max_tokens=max_vocabulary_size,
            ragged=True)

        self._source_text_vectorization.adapt(dataset.map(lambda context, target: context))
        self._target_text_vectorization.adapt(dataset.map(lambda context, target: target))

    def process_text(self, context, target):
        context = self._source_text_vectorization(context).to_tensor()
        target = self._target_text_vectorization(target)
        target_in = target[:, :-1].to_tensor()
        target_out = target[:, 1:].to_tensor()
        return (context, target_in), target_out
