import sys

import spacy

from src.dataset_utils import load_train_test_datasets
from src.utils import download_hugging_face_sentences, save_sentences_to_files

en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")


def download_and_store_sentences():
    datasets_names = ["Helsinki-NLP/europarl", "Helsinki-NLP/opus-100", "Helsinki-NLP/opus_books"]
    english_sentences, italian_sentences = download_hugging_face_sentences(datasets_names)
    save_sentences_to_files(english_sentences, italian_sentences)


def prepare_datasets():
    source_path = "datasets/english_sentences.txt"
    target_path = "datasets/italian_sentences.txt"
    train_dataset, test_dataset = load_train_test_datasets(source_path, target_path)


if __name__ == '__main__':
    sys.exit()
