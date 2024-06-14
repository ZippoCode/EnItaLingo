import tensorflow as tf
import tensorflow_text as tf_text


def tf_normalize_text(text: tf.Tensor) -> tf.Tensor:
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,-]', '')
    text = tf.strings.regex_replace(text, '[.?!,]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text
