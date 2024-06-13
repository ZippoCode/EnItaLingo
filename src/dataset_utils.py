import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_train_test_datasets(source_path: str, target_path: str, test_size=0.2, random_state=42, buffer_size=128,
                             batch_size=64):
    english_file = tf.io.gfile.GFile(source_path, 'r')
    italian_file = tf.io.gfile.GFile(target_path, 'r')

    english_sentences = english_file.read().splitlines()
    italian_sentences = italian_file.read().splitlines()

    en_train, en_test, it_train, it_test = train_test_split(english_sentences, italian_sentences, test_size=test_size,
                                                            random_state=random_state)
    train_dataset = tf.data.Dataset.from_tensor_slices((en_train, it_train)).shuffle(buffer_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((en_test, it_test)).shuffle(buffer_size).batch(batch_size)
    return train_dataset, test_dataset
