import evaluate
import numpy as np
from transformers import AutoTokenizer

from src.download_datasets import download_hugging_face_sentences, save_sentences_to_files

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
metric = evaluate.load("sacrebleu")


def build_dataset():
    english_sentences, italian_sentences = download_hugging_face_sentences()
    with open('ita.txt') as file:
        lines = file.readlines()

    for line in lines:
        english_sentences.append(line.split("\t")[0])
        italian_sentences.append(line.split("\t")[1])

    save_sentences_to_files(english_sentences, italian_sentences)
    assert len(english_sentences) == len(italian_sentences)
    print(f"Downloaded {len(english_sentences)} sentence")


def preprocess_function(examples):
    return tokenizer(examples["en"], text_target=examples["it"], max_length=128, truncation=True)


def postprocess_text(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [[label.strip()] for label in labels]

    return predictions, labels


def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions, decoded_labels = postprocess_text(decoded_predictions, decoded_labels)

    result = metric.compute(predictions=decoded_predictions, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
