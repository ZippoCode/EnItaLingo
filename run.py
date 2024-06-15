import sys

import tensorflow as tf

from src import EnitaLingo
from src.EnitaLingo import EnitaLingo
from src.dataset_utils import load_train_test_datasets
from src.utils import download_hugging_face_sentences, save_sentences_to_files


def download_and_store_sentences():
    datasets_names = ["Helsinki-NLP/europarl", "Helsinki-NLP/opus-100", "Helsinki-NLP/opus_books"]
    english_sentences, italian_sentences = download_hugging_face_sentences(datasets_names)
    save_sentences_to_files(english_sentences, italian_sentences)


def prepare_datasets(units=256):
    source_path = "datasets/english_sentences.txt"
    target_path = "datasets/italian_sentences.txt"
    train_dataset, test_dataset = load_train_test_datasets(source_path, target_path, max_sentence=1000)
    translator = EnitaLingo(units)
    translator.build_text_vectorization(dataset=train_dataset, max_vocabulary_size=5000)
    train_ds = train_dataset.map(translator.process_text, tf.data.AUTOTUNE)
    val_ds = test_dataset.map(translator.process_text, tf.data.AUTOTUNE)
    translator.build_model()


if __name__ == '__main__':
    prepare_datasets()
    sys.exit()
