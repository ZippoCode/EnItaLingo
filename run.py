from src.download_datasets import get_sentences
from src.preprocessing import build_dataset_dict
from src.train import train_model
from src.utils import preprocess_function

if __name__ == '__main__':
    english_sentences, italian_sentences = get_sentences()
    english_sentences = english_sentences[:10000]
    italian_sentences = italian_sentences[:10000]
    dataset_dict = build_dataset_dict(english_sentences, italian_sentences)
    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)
    train_model(tokenized_dataset)
