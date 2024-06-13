import os
from typing import List, Tuple

from datasets import load_dataset


def download_hugging_face_sentences(datasets_names: List[str]) -> Tuple[List[str], List[str]]:
    english_sentences = []
    italian_sentences = []
    for dataset_name in datasets_names:
        print(f"Loading {dataset_name} ...")
        dataset = load_dataset(dataset_name, name="en-it")
        translations = dataset["train"]["translation"]
        english_sentences += [sentence["en"] for sentence in translations]
        italian_sentences += [sentence["it"] for sentence in translations]
    assert len(english_sentences) == len(italian_sentences)
    print(f"Downloaded {len(english_sentences)} sentence")
    return english_sentences, italian_sentences


def save_sentences_to_files(english_sentences: List[str], italian_sentences: List[str],
                            output_dir: str = "datasets") -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "english_sentences.txt"), "w", encoding="utf-8") as f_en:
        for sentence in english_sentences:
            f_en.write(sentence + "\n")

    with open(os.path.join(output_dir, "italian_sentences.txt"), "w", encoding="utf-8") as f_it:
        for sentence in italian_sentences:
            f_it.write(sentence + "\n")
    print(f"Stored {len(english_sentences)} in {output_dir}")
    print(f"Stored {len(italian_sentences)} in {output_dir}")


def read_sentences() -> Tuple[List[str], List[str]]:
    with open(os.path.join("datasets", "english_sentences.txt"), "r", encoding="utf-8") as file:
        english_sentences = file.readlines()
    with open(os.path.join("datasets", "italian_sentences.txt"), "r", encoding="utf-8") as file:
        italian_sentences = file.readlines()
    assert len(english_sentences) == len(italian_sentences)
    print(f"Read {len(english_sentences)} sentences")
    return english_sentences, italian_sentences
