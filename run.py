import os

from dotenv import load_dotenv

from src.download_datasets import get_sentences
from src.enita_lingo_hf import train_model, predict

train = True

if __name__ == '__main__':
    load_dotenv()
    checkpoint = os.getenv('CHECKPOINT')
    checkpoint_path = os.getenv('CUSTOM_CHECKPOINT')

    english_sentences, italian_sentences = get_sentences()
    english_sentences = english_sentences[:10000]
    italian_sentences = italian_sentences[:10000]
    if train:
        train_model(english_sentences, italian_sentences)
    english_sentence = "This is a sentence because I want to learn how to build a new skill"
    italian_sentence = predict(english_sentence, checkpoint)
    print(italian_sentence)
