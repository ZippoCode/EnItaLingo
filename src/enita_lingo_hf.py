import os

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AdamWeightDecay, DataCollatorWithPadding, TFAutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
checkpoint = os.getenv('CHECKPOINT')
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
metric = evaluate.load("sacrebleu")


def preprocess_function(examples):
    return tokenizer(examples["en"], text_target=examples["it"], max_length=128, truncation=True, padding=True)


def build_dataset_dict(source_sentences, target_sentences, test_size=0.2, val_size=0.2, random_state=None):
    assert len(source_sentences) == len(target_sentences), "Source and target sentences must have the same length"
    en_train, en_test, it_train, it_test = train_test_split(source_sentences, target_sentences, test_size=test_size,
                                                            random_state=random_state)
    en_test, en_val, it_test, it_val = train_test_split(en_test, it_test, test_size=val_size / (1 - test_size),
                                                        random_state=random_state)
    train_dataset = Dataset.from_dict({"en": en_train, "it": it_train})
    val_dataset = Dataset.from_dict({"en": en_val, "it": it_val})
    test_dataset = Dataset.from_dict({"en": en_test, "it": it_test})

    dataset = DatasetDict({
        'training': train_dataset,
        'testing': test_dataset,
        'validation': val_dataset
    })

    return dataset.map(preprocess_function, batched=True)


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


def train_model(english_sentences, italian_sentences):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    tokenized_dataset = build_dataset_dict(english_sentences, italian_sentences)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tf_train_set = model.prepare_tf_dataset(
        tokenized_dataset["training"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_dataset["testing"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_dataset["validation"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
    model.compile(optimizer=optimizer)
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=10)
    model.save_pretrained('checkpoints/enita_lingo')


def predict(sentence: str, checkpoints_path: str):
    inputs = tokenizer(sentence, return_tensors="tf")["input_ids"]
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoints_path)
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
